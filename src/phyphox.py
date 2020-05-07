import os
from functools import reduce

import numpy as np
import pandas as pd

import features
import preprocessing
import shared_constants
import visualization
from file_handling import get_file_names_in_directory_for_pattern, get_parent_directory_for, get_project_directory

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


def read_experiment(experiment_path, sensors=None):
    """
    Read in the data of the experiment given by the path. Used sensors cna be adjusted by specifying the 'sensors' parameter.

    Parameters
    ----------
    experiment_path : str
        Path to file containing the sensor recordings
    sensors : array_like, optional (default=None)
        List of sensor names that should be included in the data frame. If None, all available once will be included.

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

        # sensor column name convention: {sensor_name}_{dimension}
        new_columns = list()
        for column in columns:
            new_columns.append('_'.join(column.split(' ')[:-1]).lower())
        data_frame.columns = new_columns
        data_frames.append(data_frame)

    # combine data frames and set index to a sorted pandas.TimeDeltaIndex (needed for interpolation)
    data_frame = reduce(lambda x, y: pd.merge(x, y, on='time', how='outer'), data_frames)
    data_frame = preprocessing.set_time_delta_as_index(data_frame, origin_timestamp_unit='s', output_timestamp_unit="milliseconds",
                                                       timestamp_key="time")
    return data_frame.sort_index()


def test_reading():
    experiments = get_experiments()
    data_frame = read_experiment(experiments[0])
    assert len(data_frame) == 3617
    assert data_frame.equals(data_frame.sort_index())


def test_alignment():
    experiments = get_experiments()
    data_frame = read_experiment(experiments[0])
    data_frame = preprocessing.align_data(data_frame, listening_rate=20)
    assert len(data_frame) == 1347
    assert not data_frame.isnull().values.any()


def test_visualize_experiment():
    # no real test, pytest naming
    experiments = get_experiments()
    data_frame = read_experiment(experiments[0])
    data_frame = preprocessing.align_data(data_frame, listening_rate=20)

    # split acc and gyro data for the plot
    _regex = "{sensor_name}_{dimension}".format(sensor_name="acceleration", dimension=shared_constants.DIMENSIONS_KEY_LIST)
    df1 = data_frame.filter(regex=_regex, axis=1)
    _regex = "{sensor_name}_{dimension}".format(sensor_name="gyroscope", dimension=shared_constants.DIMENSIONS_KEY_LIST)
    df2 = data_frame.filter(regex=_regex, axis=1)
    data_frames = [df1, df2]
    visualization.visualize_different_sensors(data_frames, number_of_plot_rows=2, number_of_plot_cols=1, save=True)


def test_auto_correlation():
    # no real test, pytest naming
    experiments = get_experiments()
    data_frame = read_experiment(experiments[0])
    data_frame = preprocessing.align_data(data_frame, listening_rate=20)

    # calculate auto-correlation coefficients for acceleration
    _regex = "{sensor_name}_{dimension}".format(sensor_name="acceleration", dimension=shared_constants.DIMENSIONS_KEY_LIST)
    acceleration_data_frame = data_frame.filter(regex=_regex, axis=1)
    coefficient_data_frame_acceleration = calculate_auto_correlation_data_frame(acceleration_data_frame)
    # calculate mean coefficient
    mean_coefficient_data_frame_acceleration = coefficient_data_frame_acceleration.apply(np.mean, axis=1).to_frame()

    # calculate auto-correlation coefficients for gyroscope
    _regex = "{sensor_name}_{dimension}".format(sensor_name="gyroscope", dimension=shared_constants.DIMENSIONS_KEY_LIST)
    gyroscope_data_frame = data_frame.filter(regex=_regex, axis=1)
    coefficient_data_frame_gyroscope = calculate_auto_correlation_data_frame(gyroscope_data_frame)
    mean_coefficient_data_frame_gyroscope = coefficient_data_frame_gyroscope.apply(np.mean, axis=1).to_frame()

    data_frames = [acceleration_data_frame, gyroscope_data_frame, mean_coefficient_data_frame_acceleration, mean_coefficient_data_frame_gyroscope]
    visualization.visualize_different_sensors(data_frames, number_of_plot_rows=2, number_of_plot_cols=2, save=True)

