from src.data_reading.phyphox import get_experiments, read_experiment
from src import preprocessing


def test_reading_with_merging():
    """
    Tests reading of data using multiple files that need to be aligned
    Returns
    -------

    """
    experiments = get_experiments()
    data_frame = read_experiment(experiments[1], merge_sources=True)
    assert len(data_frame) == 3617
    assert data_frame.equals(data_frame.sort_index())


def test_reading_one_file():
    """
    Tests reading of a single data file that contains both accelerometer and gyro data
    Returns
    -------

    """
    experiments = get_experiments()
    data_frame = read_experiment(experiments[0], merge_sources=False)
    assert len(data_frame) == 21871
    assert data_frame.equals(data_frame.sort_index())


def test_alignment_with_merging():
    """
    Tests aligning & interpolation of accelerometer and gyro data frame
    Returns
    -------

    """
    experiments = get_experiments()
    data_frame = read_experiment(experiments[1], merge_sources=True)
    data_frame = preprocessing.align_data(data_frame, listening_rate=20)
    assert len(data_frame) == 1347
    assert not data_frame.isnull().values.any()


def test_alignment_one_file():
    """
    Tests interpolation to custom listening rate of data from a single frame
    Returns
    -------

    """
    experiments = get_experiments()
    data_frame = read_experiment(experiments[0], merge_sources=False)
    data_frame = preprocessing.align_data(data_frame, listening_rate=20)
    assert len(data_frame) == 16294
    assert not data_frame.isnull().values.any()