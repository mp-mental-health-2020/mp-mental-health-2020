import os
from functools import reduce

import pandas as pd

from src.file_handling import get_file_names_in_directory_for_pattern, get_parent_directory_for, get_project_directory
from src.preprocessing import set_time_delta_as_index

from src.features import calculate_auto_correlation_data_frame

PHYPHOX_DIRECTORY_NAME = "phyphox"


def get_phyphox_data_directory():
    """
    It is assumed that the data directory is at the same level as the git project. If this is not the case you need to adjust the path here. The files
    are then further ordered by their respective recording app.

    Returns
    -------
        Directory containing the phyphox data
    """
    return os.path.join(get_parent_directory_for(get_project_directory()), "data", PHYPHOX_DIRECTORY_NAME)


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


def read_experiment(experiment_path, sensors=None, merge_sources=False):
    """
    Read in the data of the experiment given by the path. Used sensors cna be adjusted by specifying the 'sensors' parameter.

    Parameters
    ----------
    experiment_path : str
        Path to file containing the sensor recordings
    sensors : array_like, optional (default=None)
        List of sensor names that should be included in the data frame. If None, all available once will be included.
    merge_sources: bool
        True, if we have separate accelerometer and gyro files that need merging. Default False, as we're using the new phyphox configuration
    Returns
    -------
        pd.DataFrame for all (specified) sensors with an sorted pd.TimeDeltaIndex. May contain 'NaN' values if sensors are not in sync.
    """
    file_names = get_file_names_in_directory_for_pattern(experiment_path, "*.csv")
    if sensors:
        file_names = filter_files(file_names, sensors)

    data_frames = list()
    for file_name in file_names:
        data_frame = pd.read_csv(file_name)
        columns = data_frame.columns

        # for our old data samples we need to rename the columns
        if merge_sources:
            # sensor column name convention: {sensor_name}_{dimension}
            new_columns = list()
            for column in columns:
                new_columns.append('_'.join(column.split(' ')[:-1]).lower())
            data_frame.columns = new_columns
        data_frames.append(data_frame)

    # combine data frames and set index to a sorted pandas.TimeDeltaIndex (needed for interpolation)
    timestamp_column_name = "time" if merge_sources else "timestamp"
    if merge_sources:
        data_frame = reduce(lambda x, y: pd.merge(x, y, on=timestamp_column_name, how='outer'), data_frames)
    else:
        data_frame = data_frames[0]
    data_frame = set_time_delta_as_index(data_frame, origin_timestamp_unit='s',
                                                           output_timestamp_unit="milliseconds",
                                                           timestamp_key=timestamp_column_name)
    return data_frame.sort_index()
