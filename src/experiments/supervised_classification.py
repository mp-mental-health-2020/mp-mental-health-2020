import pandas as pd
from sklearn.preprocessing import StandardScaler
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute

from classification.classification import classify_all
from data_reading.phyphox import read_experiments_in_dir
from features import extract_timeseries_features
from file_handling import get_sub_directories
from preprocessing import concat_chunks_for_feature_extraction, preprocess_chunks_for_null_test, preprocess_chunks_for_null_test_with_indoor, \
    segment_null_classification


def run_supervised_classification(experiment_dir_path):
    experiment_dirs = get_sub_directories(experiment_dir_path)

    # Read data
    use_indoor = True
    sample_rate = 50
    chunks, null_chunks, y = read_experiments_in_dir(experiment_dirs, sample_rate, drop_lin_acc=False, require_indoor=use_indoor)

    del experiment_dir_path
    del experiment_dirs

    # TODO: add assertions
    print("Finished reading data")

    # we only need the y vector for the multi class clf
    y.reset_index(inplace=True)

    # Preprocess data

    if use_indoor:
        chunks_ocd, chunks_null_class = preprocess_chunks_for_null_test_with_indoor(chunks, null_chunks)
    else:
        chunks_ocd, chunks_null_class = preprocess_chunks_for_null_test(chunks, null_chunks)

    del chunks
    del null_chunks
    # Segmentation

    window_size = 50  # 1 second
    chunks_ocd_segmented, labels_ocd_segmented, chunks_null_segmented, labels_null_segmented = segment_null_classification(chunks_ocd,
                                                                                                                           chunks_null_class,
                                                                                                                           window_size)
    del chunks_ocd
    del chunks_null_class

    # TODO: remove this
    chunks_ocd_segmented = chunks_ocd_segmented[:20]
    labels_ocd_segmented = labels_ocd_segmented[:20]
    chunks_null_segmented = chunks_null_segmented[:20]
    labels_null_segmented = labels_null_segmented[:20]

    # for the feature extraction we need all of our data in one concatenated df - tsfresh groups by segment id
    null_classification_df, labels_null_classification = concat_chunks_for_feature_extraction(
        [chunks_ocd_segmented, chunks_null_segmented],
        [labels_ocd_segmented, labels_null_segmented])
    assert len(set(labels_null_classification)) == 2

    del chunks_ocd_segmented
    del chunks_null_segmented
    del labels_ocd_segmented
    del labels_null_segmented

    print("Finished data preparation and segmentation")
    # Feature extraction

    action_ids = null_classification_df["action_id"].values

    X_null_classification = extract_timeseries_features(null_classification_df)

    print("Finished feature extraction")

    # Feature selection
    impute(X_null_classification)
    X_null_classification_selected = select_features(X_null_classification, labels_null_classification)

    print("Finished feature selection")

    scaler = StandardScaler()
    X_null_classification_scaled = scaler.fit_transform(X_null_classification_selected)

    # Classification
    classify_all(X_null_classification_scaled, labels_null_classification)


def test_run_short_recordings_clf():
    run_supervised_classification("../../data/phyphox/full recordings/")


def test_df():
    df_1 = pd.DataFrame(data=[[1, 2, 5], [3, 4, 6]], columns=["a", "b", "c"])
    df_2 = df_1.loc[:, ["a", "c"]]
    print()
