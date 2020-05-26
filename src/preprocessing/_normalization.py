from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import check_is_fitted


def normalize_using_min_max_scaling(data_frame, normalizer=None, feature_range=(0, 1), return_normalizer=False):
    """
    Used to adjust values measured on different scales to a notionally common scale

    Parameters
    ----------
    data_frame : pd.DataFrame
        DataFrame containing the data to be normalized.
    normalizer : sklearn.preprocessing.MinMaxScaler, default=None
        Predefined scaler to be used over multiple iterations.
    feature_range : tuple, default=(0, 1)
        Range in which the normalized values should be returned.
    return_normalizer : bool, default=False
        If true, the used scaler will be returned.
    Returns
    -------
        DataFrame containing normalized columns. If return_normalizer is true it will also return the scaler used for the transformation.
    """
    if not normalizer:
        normalizer = MinMaxScaler(feature_range=feature_range)

    try:
        check_is_fitted(normalizer)
    except NotFittedError:
        normalizer.fit(data_frame)

    if return_normalizer:
        return normalizer.transform(data_frame), normalizer
    return normalizer.transform(data_frame)
