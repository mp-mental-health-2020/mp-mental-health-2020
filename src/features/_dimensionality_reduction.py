import pandas as pd
from sklearn.decomposition import PCA

from src.data_reading.phyphox import get_random_aligned_test_file
from src.preprocessing import normalize_using_min_max_scaling


def transform_data_using_pca(data_frame, pca=None, return_pca=False, normalize_data=True, svd_solver='full', whiten=False, n_components=0.95,
                             random_state=None):
    """
    Transforms the given data using a Principal Component Analysis (PCA). If a pca is given, components will not be be calculated on the given
    data_frame but the given once will be used. This method can be used to reduce the dimensionality of the given data_frame. In order to do that
    the n_components parameter must be provided.

    Parameters
    ----------
    data_frame : pd.DataFrame
        DataFrame containing the data you want to transform.
    pca : sklearn.PCA, default=None
        Already fitted sklearn.PCA to transform the given data with the supplied principal components.
    return_pca : bool, default=False
        If True, the fitted (or given) pca will be returned.
    svd_solver : str {'auto', 'full', 'arpack', 'randomized'}
        If auto :
            The solver is selected by a default policy based on `X.shape` and
            `n_components`: if the input data is larger than 500x500 and the
            number of components to extract is lower than 80% of the smallest
            dimension of the data, then the more efficient 'randomized'
            method is enabled. Otherwise the exact full SVD is computed and
            optionally truncated afterwards.
        If full :
            run exact full SVD calling the standard LAPACK solver via
            `scipy.linalg.svd` and select the components by postprocessing
        If arpack :
            run SVD truncated to n_components calling ARPACK solver via
            `scipy.sparse.linalg.svds`. It requires strictly
            0 < n_components < min(X.shape)
        If randomized :
            run randomized SVD by the method of Halko et al.
    whiten : bool, optional (default False)
        When True (False by default) the `components_` vectors are multiplied
        by the square root of n_samples and then divided by the singular values
        to ensure uncorrelated outputs with unit component-wise variances.

        Whitening will remove some information from the transformed signal
        (the relative variance scales of the components) but can sometime
        improve the predictive accuracy of the downstream estimators by
        making their data respect some hard-wired assumptions.
    n_components : int, default=0.95
        The estimated number of components. When n_components is set to 'mle' or a number between 0 and 1 (with svd_solver == 'full') this number is
        estimated from input data. Otherwise it equals the parameter n_components, or the lesser value of n_features and n_sample if n_components
        is None. If the value is 'mle' or between 0 and 1, than the number of dimensions is chosen by how many dimensions are needed to explain at
        least that amount of variance in the data.

    Returns
    -------
    transformed_data_frame : pd.DataFrame
        Data frame with transformed data and (maybe) reduced dimensionality.
    pca : sklearn.PCA (optional)
        Fitted principal component analysis if the parameter return_pca is True.
    """
    if not pca:
        pca = PCA(svd_solver=svd_solver, whiten=whiten, n_components=n_components, random_state=random_state)
        pca.fit(data_frame)

    if normalize_data:
        # This is done to ensure similar value ranges for our data as this could affect the variance evaluation for our data
        data_frame = normalize_using_min_max_scaling(data_frame)

    # Note that a pca needs centered data, but the pca will take care of it by itself
    transformed_data_frame = pd.DataFrame(pca.transform(data_frame))
    if return_pca:
        return transformed_data_frame, pca
    return transformed_data_frame


def test_usage_of_method():
    data_frame = get_random_aligned_test_file()
    transformed_data_frame = transform_data_using_pca(data_frame, n_components=2)
    # you now have a data frame with 2 columns and transformed data
