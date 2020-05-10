import numpy as np

from src import preprocessing, visualization, shared_constants
from src.data_reading.phyphox import get_experiments, read_experiment
from src.features import calculate_auto_correlation_data_frame


def autocorrelation_phyphox():
    experiments = get_experiments()
    data_frame = read_experiment(experiments[0])
    data_frame = preprocessing.align_data(data_frame, listening_rate=20)

    # calculate auto-correlation coefficients for acceleration
    _regex = "{sensor_name}_{dimension}".format(sensor_name="acceleration",
                                                dimension=shared_constants.DIMENSIONS_KEY_LIST)
    acceleration_data_frame = data_frame.filter(regex=_regex, axis=1)
    coefficient_data_frame_acceleration = calculate_auto_correlation_data_frame(acceleration_data_frame)
    # calculate mean coefficient
    mean_coefficient_data_frame_acceleration = coefficient_data_frame_acceleration.apply(np.mean, axis=1).to_frame()

    # calculate auto-correlation coefficients for gyroscope
    _regex = "{sensor_name}_{dimension}".format(sensor_name="gyroscope", dimension=shared_constants.DIMENSIONS_KEY_LIST)
    gyroscope_data_frame = data_frame.filter(regex=_regex, axis=1)
    coefficient_data_frame_gyroscope = calculate_auto_correlation_data_frame(gyroscope_data_frame)
    mean_coefficient_data_frame_gyroscope = coefficient_data_frame_gyroscope.apply(np.mean, axis=1).to_frame()

    data_frames = [acceleration_data_frame, gyroscope_data_frame, mean_coefficient_data_frame_acceleration,
                   mean_coefficient_data_frame_gyroscope]
    visualization.visualize_different_sensors(data_frames, number_of_plot_rows=2, number_of_plot_cols=2, save=True)


if __name__ == "__main__":
    # execute only if run as a script
    autocorrelation_phyphox()
