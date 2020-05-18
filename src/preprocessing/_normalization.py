def normalize_using_min_max_scaling(data_frame):
    """
    Used to adjust values measured on different scales to a notionally common scale

    Parameters
    ----------
    data_frame : pd.DataFrame
        DataFrame containing the data to be normalized.

    Returns
    -------
        DataFrame containing normalized columns.
    """
    max_value_vector = data_frame.apply(max)
    min_value_vector = data_frame.apply(min)
    return data_frame.apply(lambda x: ((x - min_value_vector) / (max_value_vector - min_value_vector) - 0.5) * 2, axis=1)
