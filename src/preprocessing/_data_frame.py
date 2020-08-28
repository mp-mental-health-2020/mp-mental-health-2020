import pandas as pd

from src import shared_constants


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


def remove_user_from_df(X):
    """
    Removes the 'user' column from the df of features and returns it.
    This is needed for LOOCV to later reference which entry belongs to which user.
    Since the order of entries doesn't change it is enough to return the single column.
    Parameters
    ----------
    X

    Returns the user column
    -------

    """
    user_col = X.loc[:, ["user", "combined_id"]].groupby("combined_id", as_index = False).median()
    X.drop("user",axis=1, inplace=True)
    X.drop("user_right",axis=1, inplace=True)

    # return the most frequent user value for each segment (even though the value should never differ within one segment)
    return user_col["user"]
