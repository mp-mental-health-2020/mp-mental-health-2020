from src.data_reading.phyphox import get_experiments, read_experiment
from src import preprocessing


def test_reading():
    experiments = get_experiments()
    data_frame = read_experiment(experiments[0])
    assert len(data_frame) == 3617
    assert data_frame.equals(data_frame.sort_index())


def test_alignment():
    experiments = get_experiments()
    data_frame = read_experiment(experiments[0])
    data_frame = preprocessing.align_data(data_frame, listening_rate=20)
    assert len(data_frame) == 1347
    assert not data_frame.isnull().values.any()
