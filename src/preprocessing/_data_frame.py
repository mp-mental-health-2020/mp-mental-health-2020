import pandas as pd

import src.shared_constants as shared_constants


def set_time_delta_as_index(data_frame, origin_timestamp_unit="ms", output_timestamp_unit="milliseconds", timestamp_key=None):
    """
    Set timestamp column, converted to timeDeltaIndex, as index.

    Parameters
    ----------
    data_frame : pd.DataFrame
        pd.DataFrame containing the data.
    origin_timestamp_unit : string, default=‘ms’
        With unit=’ms’ and origin=’unix’ (the default), this would calculate the number of milliseconds to the unix epoch start. This is the time
        unit the data in the 'timestamp_key' column are given in.
    output_timestamp_unit : string, default=‘milliseconds’
        Time unit in which the 'pd.TimedeltaIndex' will be presented. Note that 'ms' is not a valid option here and that the timestamps in the
        index are visualized in 'nanoseconds' regardless of this value as they are converted to a 'timedelta64[ns]' dtype.
    timestamp_key : str, default=None
        Key of the column the time data is in. If not provided, default key will be used.

    Returns
    -------
    original_data_frame : pd.DataFrame
        pd.DataFrame using a 'pd.TimeDeltaIndex' with dropped original column.
    """
    if not timestamp_key:
        timestamp_key = shared_constants.TIMESTAMP_KEY
    timestamp_to_date = pd.to_datetime(data_frame.loc[:, timestamp_key], unit=origin_timestamp_unit)
    time_delta_index = pd.TimedeltaIndex(timestamp_to_date, unit=output_timestamp_unit)
    data_frame = pd.DataFrame(data_frame.values, index=time_delta_index, columns=data_frame.columns)
    data_frame.drop(timestamp_key, axis=1, inplace=True)
    return data_frame
