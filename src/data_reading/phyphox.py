import os
from random import sample

import pandas as pd

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


def read_experiment(experiment_path, sensors=None, offsets=None):
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
        # we either have multiple data frames for different hands or just one that we can return right away
        if hand:
            data_frames[hand] = data_frame
        else:
            return data_frame
    return data_frames


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
