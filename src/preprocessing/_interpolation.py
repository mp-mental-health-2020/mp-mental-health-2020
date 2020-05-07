import pandas as pd
from scipy import interpolate

import src.shared_constants as shared_constants
from src.exceptions import IncorrectInputTypeException


def align_data(data_frame, interpolation_method='linear', listening_rate=20, reference_sensor="acceleration"):
    if not isinstance(data_frame, pd.DataFrame):
        raise IncorrectInputTypeException(data_frame, pd.DataFrame)

    # interpolate reference sensor
    _regex = "{sensor_name}_{dimension}".format(sensor_name=reference_sensor, dimension=shared_constants.DIMENSIONS_KEY_LIST)
    reference_sensor_data = data_frame.filter(regex=_regex, axis=1)

    interpolated_reference_sensor = correct_sampling_frequency_by_interpolation(data_frame=reference_sensor_data.dropna(how='all'),
                                                                                interpolation_method=interpolation_method,
                                                                                listening_rate=listening_rate,
                                                                                time_unit='ms')
    interpolation_data_list = [interpolated_reference_sensor]

    # interpolate other sensors
    for sensor in data_frame.filter(regex=".*_(x|0)", axis=1).columns:
        sensor = sensor.split("_")[0]
        if sensor == reference_sensor:
            continue

        _regex = "{sensor_name}_{dimension}".format(sensor_name=sensor, dimension=shared_constants.DIMENSIONS_KEY_LIST)
        sensor_data = data_frame.filter(regex=_regex, axis=1)
        interpolated_sensor_data = correct_sampling_frequency_by_interpolation(data_frame=sensor_data.dropna(how='all'),
                                                                               interpolation_method=interpolation_method,
                                                                               listening_rate=listening_rate,
                                                                               time_unit='ms',
                                                                               start=interpolated_reference_sensor.index[0],
                                                                               end=interpolated_reference_sensor.index[-1])
        interpolation_data_list.append(interpolated_sensor_data)
    return pd.concat(interpolation_data_list, axis=1)


def correct_sampling_frequency_by_interpolation(data_frame, interpolation_method="linear", listening_rate=20, time_unit='ms', start=None, end=None):
    """
    Parameters
    ----------
    data_frame : pd.DataFrame
        The data used to interpolate along the features' axis; two-dimensional data frame, shape (n_samples, n_features), index is a TimeDeltaIndex
        of time series' timestamp information.
    interpolation_method : str, default="linear"
        Specification which interpolation method should be used.
    listening_rate : int, default=20
        Number of milliseconds between two time events; it is used for sample frequency correction, pretending that sensor events are called
        consistent in a interval of e.g. 20 milliseconds.
    time_unit : str, default='ms' (milliseconds)
        Time unit the 'pd.TimeDeltaIndex' of the original_data_frame is given in.
    start : pd.TimedeltaIndex, default=None
        TimedeltaIndex where to start with interpolation. If not provided, the earliest timestamp of any sensor will be used.
    end : pd.TimedeltaIndex, default_None
        TimedeltaIndex where to end with interpolation. If not provided, the latest timestamp of any sensor will be used.

    Returns
    -------
    interpolated_data_frame : pd.DataFrame
    """
    if not isinstance(data_frame.index, pd.TimedeltaIndex):
        raise Exception("The index of the data frame is not of type timeDelta. "
                        "Treating the given index values as milliseconds would result in wrong results.")

    if start is None:
        start = data_frame.index[0]
    if end is None:
        end = data_frame.index[-1]

    time_delta = pd.Timedelta(value=listening_rate, unit=time_unit)
    new_index = pd.timedelta_range(start=start, end=end, freq=time_delta)
    interpolation_function = interpolate.interp1d(data_frame.index.astype('i8'), data_frame.values.T, kind=interpolation_method,
                                                  fill_value="extrapolate")
    interpolated_data_frame = pd.DataFrame(data=interpolation_function(new_index.astype('i8')).T, index=new_index)
    interpolated_data_frame.columns = data_frame.columns

    return interpolated_data_frame
