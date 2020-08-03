# this file is making live predictions if a sample is of an OCD activity or not
from sklearn.preprocessing import StandardScaler
from tsfresh import select_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from classification.classification import train_and_select_best_model, predict
from data_reading.phyphox import read_experiments_in_dir
from features import extract_timeseries_features
from file_handling import get_sub_directories
from preprocessing import segment_for_null_classification, segment_windows, concat_chunks_for_feature_extraction, \
    preprocess_chunks_for_null_test
import pandas as pd
from shared_constants import SEGMENTATION_NO_OVERLAP

sample_rate = 50

experiment_dir_path = "../../data/phyphox/full recordings/"
experiment_dirs = get_sub_directories(experiment_dir_path)
use_indoor = True
use_fingerprinting_approach = True
window_size = 100
feature_calculation_setting = MinimalFCParameters()

experiment_dirs_selected = ["Ana-2","Anne","Ariane","Cilly","Fabi","Julian","Julius","Wiktoria"]
experiment_dirs = [exp_dir for exp_dir in experiment_dirs if exp_dir.split("/")[-1] in experiment_dirs_selected]

selected_activities = ["washing hands", "drying hands"]

# Read data
chunks, null_chunks, y = read_experiments_in_dir(experiment_dirs, sample_rate, drop_lin_acc=True,
                                                 require_indoor=use_indoor, selected_activities=selected_activities)

del experiment_dirs
print("Finished reading data")

chunks_ocd, chunks_null_class = preprocess_chunks_for_null_test(chunks, null_chunks, use_indoor=use_indoor)
#labels = y_ocd.loc[:, "label"].squeeze()
#assert len(labels) == len(chunks_ocd)
del chunks
del null_chunks
chunks_ocd_segmented, labels_ocd_segmented, chunks_null_segmented, labels_null_segmented = segment_for_null_classification(chunks_ocd, chunks_null_class, window_size)

assert len(set(labels_ocd_segmented)) == 1
assert len(set(labels_null_segmented)) == 1

null_classification_df, labels_null_classification = concat_chunks_for_feature_extraction(
        [chunks_ocd_segmented, chunks_null_segmented],
        [labels_ocd_segmented, labels_null_segmented])
assert len(set(labels_null_classification)) == 2

X_null_class_classification = extract_timeseries_features(null_classification_df, use_indoor=use_indoor,
                                                              feature_set_config=feature_calculation_setting,
                                                              use_fingerprinting_approach=use_fingerprinting_approach)
impute(X_null_class_classification)
X_null_classification_selected = select_features(X_null_class_classification, labels_null_classification)

# store the features so that we can apply the same feature selection later on the test data
selected_features = X_null_classification_selected.columns

scaler = StandardScaler()
X_null_classification_scaled = scaler.fit_transform(X_null_classification_selected)

trained_model = train_and_select_best_model(X_null_classification_scaled, labels_null_classification)


# test on long recording

test_data_dir = experiment_dir_path + "Marvin/"
chunks_test, null_chunks_test, y_test = read_experiments_in_dir([test_data_dir], sample_rate, drop_lin_acc=True,
                                                 require_indoor=use_indoor)

# we need to zip the chunks back
chunks_ocd_test, chunks_null_class_test = preprocess_chunks_for_null_test(chunks_test, null_chunks_test, use_indoor=use_indoor)

# we need to zip the chunks back
chunks_test_all = list(sum(zip(chunks_ocd_test, chunks_null_class_test),())) # we might need to append the last element from the chunks_test at the end as well

chunks_test_segmented, labels_test_segmented = segment_windows(chunks_test_all, ["Test"] * len(chunks_test_all), window_size, SEGMENTATION_NO_OVERLAP)

# split into blocks of 10 chunks for which we want to predict
block_size = 10
segment_id = 0
blocks = [chunks_test_segmented[i*block_size:(i+1)*block_size] for i in range(int(len(chunks_test_segmented)/block_size))]
for b in blocks:
    current_df, _ = concat_chunks_for_feature_extraction(chunks=[b], labels=[pd.Series(["Test"] * block_size)]) # the labels don't matter
    X_test = extract_timeseries_features(current_df, use_indoor=use_indoor,
                                                              feature_set_config=feature_calculation_setting,
                                                              use_fingerprinting_approach=use_fingerprinting_approach)
    impute(X_test)
    X_test = X_test.loc[:, list(selected_features)]
    X_test_scaled = scaler.transform(X_test)
    predictions = predict(X=X_test, model=trained_model)
    for i in range(block_size):
        print("Action: {}: start time: {}: {}".format(b[i]["action_id"][0], b[i].reset_index()["index"][0].total_seconds(), predictions[i]))

# classifier = train multiclass clf
# test_segmented = segment_windows(...)
# for each in test_segmented:
#  classifier.predict(...)