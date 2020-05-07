import pandas as pd


def apply_moving_average_filter(series: pd.Series, window_size):
    """
    Apply moving average filter to a pandas series.

    Parameters
    ----------
    series : pd.Series
        Series to be filtered
    window_size : int
        Number of values to be used for the average function.

    Returns
    -------
        Smoothened data series.
    """
    return series.rolling(window=window_size, min_periods=1).mean()
