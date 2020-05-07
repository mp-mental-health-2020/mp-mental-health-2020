import numpy as np
import pandas as pd


def calculate_auto_correlation_coefficients(series):
    """
    The auto-correlation function from  Box and Jenkins (1976).
    Parameters
    ----------
    series : pd.Series
        Series for which the correlation will be calculated.

    Returns
    -------
    pd.Series
        Returns auto-correlation coefficients for the given series with the index being the lag used and the value being the corresponding
        correlation.
    """
    values = series.values
    coefficients = list()
    number_of_items = len(series)
    mean_elem = series.mean()
    for t in range(number_of_items):
        numerator = sum([(values[index] - mean_elem) * (values[index + t] - mean_elem) for index in range(number_of_items - t)])
        dividend = (series - mean_elem).apply(np.square).sum()
        coefficient = numerator / dividend
        coefficients.append(coefficient)
    return pd.Series(coefficients)


def calculate_auto_correlation_data_frame(original_data_frame, print_most_influencing_indices=False, number_of_ignored_lags=10, number_of_indices=3):
    """
    Calculates the auto-correlation for every column of the data.

    Parameters
    ----------
    original_data_frame : pd.DataFrame
        Data Frame containing the data for which the correlation will be calculated.
    print_most_influencing_indices : bool, default=False
        If True, the highest #number_of_indices lags will be printed for each column.
    number_of_ignored_lags : int, default=10
        Number of lags to ignore at the start and end as lag=0 will have a correlation of 1.
    number_of_indices : int, default=3
        Number of highest coefficients to print.

    Returns
    -------
    pd.DataFrame
        Data frame containing the auto-correlation coefficients for their respective columns.
    """
    auto_correlation_coefficients = list()
    for name, column in original_data_frame.iteritems():
        coefficients = calculate_auto_correlation_coefficients(column)
        auto_correlation_coefficients.append(coefficients)

        if print_most_influencing_indices:
            coefficients = coefficients.iloc[number_of_ignored_lags:-number_of_ignored_lags]
            coefficients = coefficients.nlargest(number_of_indices, keep='all')
            print("\nTop {count} coefficients for {name}: {coef}".format(count=number_of_indices, name=name, coef=coefficients))

    resulting_data_frame = pd.DataFrame(auto_correlation_coefficients).T
    resulting_data_frame.columns = original_data_frame.columns
    return resulting_data_frame
