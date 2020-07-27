import os
from random import sample

import pandas as pd

import file_handling
from indoor_positioning import get_file_as_data_frame, get_prepared_beacon_data
from indoor_positioning.constant import BEACON_MINORS
from preprocessing import align_data
from src import preprocessing
from src.file_handling import get_file_names_in_directory_for_pattern, get_project_directory
from src.preprocessing import set_time_delta_as_index

PHYPHOX_DIRECTORY_NAME = "phyphox"


def get_phyphox_data_directory():
    """
    It is assumed that the data directory is at the same level as the git project. If this is not the case you need to adjust the path here. The files
    are then further ordered by their respective recording app.

    Returns
    -------
        Directory containing the phyphox data
    """
    return os.path.join(get_project_directory(), "data", PHYPHOX_DIRECTORY_NAME)


def get_experiments():
    """
    Retrieve all available experiment files for the phyphox data directory.

    Returns
    -------
        List of file paths to the experiments containing the data files.
    """
    directory = get_phyphox_data_directory()
    experiments = list()
    for f in os.listdir(directory):
        file_name = os.path.join(directory, f)
        if os.path.isdir(file_name):
            experiments.append(file_name)

    return experiments


def filter_files(file_names, sensors):
    filtered_files = list()
    for file_name in file_names:
        sensor = file_name.split("/")[-1].split(".")[0]
        if sensor.lower() in sensors:
            filtered_files.append(file_name)

    return filtered_files


def read_experiment(experiment_path, sensors=None, offsets=None, drop_lin_acc=True):
    """
    Read in the data of the experiment given by the path. Used sensors cna be adjusted by specifying the 'sensors' parameter.

    Parameters
    ----------
    experiment_path : str
        Path to file containing the sensor recordings
    sensors : array_like, optional (default=None)
        List of sensor names that should be included in the data frame. If None, all available ones will be included.
    offsets: dict or None
        if dict then we provide the offsets for each file - the last part of the file name will be associated with the offsets for each hand
    drop_lin_acc: bool that indicates if the linear acceleration should be considered
    Returns
    -------
        pd.DataFrame for all (specified) sensors with an sorted pd.TimeDeltaIndex. May contain 'NaN' values if sensors are not in sync.
    """
    file_names = get_file_names_in_directory_for_pattern(experiment_path, "*.csv")
    if sensors:
        file_names = filter_files(file_names, sensors)

    timestamp_column_name = "timestamp"
    data_frames = {}
    for file_name in file_names:
        data_frame = pd.read_csv(file_name)

        # drop the linear acceleration columns here
        if drop_lin_acc:
            data_frame.drop(columns=["linear_acceleration x", "linear_acceleration y", "linear_acceleration z"], inplace=True)
            # columns: timestamp, 3 x acceleration, 3 x gyro
            assert len(data_frame.columns) == 7

        offset = None
        hand = file_name.replace(".csv", "").split("_")[-1]
        hand = hand if hand in ["left", "right"] else None
        if offsets:
            hand = file_name.replace(".csv", "").split("_")[-1]
            if hand:
                offset = offsets[hand]
            else:
                raise ValueError("no offset specified for the given hand", hand)

        if offset:
            offset_index = data_frame.iloc[(data_frame[timestamp_column_name] - float(offset)).abs().argsort()[:1]].index.tolist()[0]
            data_frame = data_frame.iloc[offset_index:, :]
        data_frame = set_time_delta_as_index(data_frame, origin_timestamp_unit='s',
                                             output_timestamp_unit="milliseconds",
                                             timestamp_key=timestamp_column_name)
        data_frame.sort_index(inplace=True)
        if hand:
            data_frames[hand] = data_frame
    return data_frames


def read_experiments_in_dir(experiment_dirs, sample_rate, drop_lin_acc=True, require_indoor=True):
    """

    Parameters
    ----------
    experiment_dirs : array-like
        List of all recording subdirectories which should be used for the current run.
    sample_rate : int
        Sample rate used for interpolating the data.
    drop_lin_acc : boolean, default=True
        If True, linear acceleration is not used for this run.
    require_indoor : boolean, default=True
        If True, it requires the recording directories to contain indoor positioning recordings.

    Returns
    -------
        chunks : dict
            Contains keys for the data used:
                'left' and 'right' for the sensor data of the respective hands
                'indoor' for the beacon advertising data
            Each value contains a lists of data frames for the respective times when activities where performed in the recording. Each list entry
            is one activity.
        null_chucks : dict
            Same as 'chunks' but with the data for when no activity was performed.
        y : pd.DataFrame
            DataFrame containing the labels for the activities performed and their respective hands.
    """
    chunks = {"right": [], "left": []}
    null_chunks = {"right": [], "left": []}
    if require_indoor:
        chunks["indoor"] = []
        null_chunks["indoor"] = []
    y_columns = ["start", "end", "label", "hand"]
    y = pd.DataFrame(columns=y_columns)

    for directory in experiment_dirs:
        offsets = {}
        try:
            with open(directory + "/offset.txt") as f:
                for line in f:
                    (key, val) = line.split(": ")
                    offsets[key.lower()] = val
        except FileNotFoundError:
            print("No offset file available in {}".format(directory))
            continue

        data_frames = read_experiment(directory, offsets=offsets, drop_lin_acc=drop_lin_acc)

        data_frames = {key: align_data(data_frame, listening_rate=1000 / sample_rate, reference_sensor=None) for
                       key, data_frame in data_frames.items()}

        if require_indoor:
            indoor_data = get_indoor_data(directory, sample_rate, offsets=offsets)
            if indoor_data is None:
                continue
            data_frames["indoor"] = indoor_data

        # get relevant data from action annotations
        y_actions = pd.read_csv(directory + "/annotations.tsv", delimiter="\t", header=None)
        y_actions = y_actions.iloc[:, [3, 5, 8]]

        # get relevant data from hand annotations
        y_hands = pd.read_csv(directory + "/hands.tsv", delimiter="\t", header=None)
        y_hands = y_hands.iloc[:, [8]]

        # combine labels
        y_current = pd.concat([y_actions, y_hands], axis=1)
        y_current.columns = y_columns

        y = y.append(y_current)

        # iterate over the annotations and split the timeseries in chunks
        for key, df in data_frames.items():
            if key in chunks:
                chunks[key] += [df.iloc[int(annotation["start"] * sample_rate):int(annotation["end"] * sample_rate)] for i, annotation in
                                y_current.iterrows()]
                # null chunks are everything in between annotations
                null_chunks[key] += [df.iloc[int(annotation["end"] * sample_rate):int(y_current.iloc[i + 1:i + 2]["start"] * sample_rate)] for
                                     i, annotation in y_current.iterrows() if i < len(y_current) - 1]

    # clear all the labels with multiple consecutive white spaces
    y["label"] = y["label"].str.replace("  ", " ").str.strip()
    y.reset_index(inplace=True, drop=True)
    # we have to do this twice to access the index column using .loc
    y.reset_index(inplace=True)
    # make sure that in the labels vector we have no duplicate indexes
    assert len(y.loc[:, "index"].unique()) == len(y)
    return chunks, null_chunks, y


def get_indoor_data(directory, sample_rate, offsets=None):
    """
    TODO 24.07.2020: outdated documentation
    Gets the aggregated indoor positioning data from the given directory with synchronization performed, if given. Data will be aggregated to only
    return the strongest beacon data in each chunk. This data will be interpolated to be at the given sample rate. Will return 'None' if no indoor
    data could be found within the directory.

    Parameters
    ----------
    directory : str
        Path to the directory containing the data.
    sample_rate : int
        Requested frequency of data after interpolation in Hz.
    offsets : dict, default=None
        If supplied and it contains the 'indoor' key, the value will be used to remove the first few seconds of data for synchronization purposes.
        Value is given in seconds.

    Returns
    -------
        If data is available, returns pd.DataFrame with synchronized, aggregated and interpolated indoor positioning data. If not, returns None.
    """
    try:
        # long recordings are split into multiple files, as energy saving options of some smart-phones will terminate bluetooth scanning without
        # notification making it impossible to handle it in app. An threshold based on experience is 30 minutes.
        indoor_files = file_handling.get_file_names_in_directory_for_pattern(directory, "*.json")

        indoor_data_frame_structure = dict()
        indoor_data_frame_structure["timestamp"] = []

        for beacon_id in BEACON_MINORS:
            indoor_data_frame_structure[beacon_id] = []

        indoor_data_frame = pd.DataFrame(indoor_data_frame_structure)

        for indoor_file in indoor_files:
            current_indoor_data_frame = get_file_as_data_frame(indoor_file)

            # synchronization
            if offsets and "indoor" in offsets.keys():
                offset = float(offsets["indoor"])
                start_timestamp = current_indoor_data_frame["timestamp"][0]
                # offset is save in seconds
                synchronized_start_timestamp = start_timestamp + offset * 1000
                current_indoor_data_frame = current_indoor_data_frame[current_indoor_data_frame["timestamp"] >= synchronized_start_timestamp]

            current_indoor_data_frame = get_prepared_beacon_data(current_indoor_data_frame)
            indoor_data_frame = indoor_data_frame.append(current_indoor_data_frame, ignore_index=True)

        indoor_data_frame = set_time_delta_as_index(indoor_data_frame, origin_timestamp_unit='ms',
                                                    output_timestamp_unit="milliseconds",
                                                    timestamp_key="timestamp")
        indoor_data_frame.sort_index(inplace=True)

        return align_data(indoor_data_frame, interpolation_method="previous", listening_rate=1000 / sample_rate,
                          reference_sensor=None)
    except IndexError:
        # we don't have an indoor recording for this recording session
        return None


def get_random_aligned_test_file():
    """
    This method will return a random pyphox experiment to use it while developing. This is a convenience method for developing.
    Returns
    -------
        Usable data frame with aligned sensor data and without 'Nan' values.
    """
    experiments = get_experiments()
    data_frame = read_experiment(sample(experiments, 1)[0])
    return preprocessing.align_data(data_frame, listening_rate=20)
