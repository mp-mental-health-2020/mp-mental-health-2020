# this file is supposed to summarize our results with respect to OCD recognition
from tsfresh import select_features
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from data_reading.phyphox import read_experiments_in_dir
from features import extract_timeseries_features
from file_handling import get_sub_directories
from preprocessing import preprocess_chunks_for_multiclass_test_one_handed, \
    segment_for_null_classification, segment_windows, concat_chunks_for_feature_extraction

sample_rate = 50

experiment_dir_path = "../../data/phyphox/full recordings/"
experiment_dirs = get_sub_directories(experiment_dir_path)
use_indoor = True
use_fingerprinting_approach = True
window_size = 50
feature_calculation_setting = MinimalFCParameters()

experiment_dirs_selected = ["Ana-2","Anne","Ariane","Cilly","Fabi","Julian","Julius","Wiktoria"]
experiment_dirs = [exp_dir for exp_dir in experiment_dirs if exp_dir.split("/")[-1] in experiment_dirs_selected]

# Read data
chunks, null_chunks, y = read_experiments_in_dir(experiment_dirs, sample_rate, drop_lin_acc=True,
                                                 require_indoor=use_indoor,
                                                 use_fingerprinting_approach=use_fingerprinting_approach)

del experiment_dir_path
del experiment_dirs
print("Finished reading data")

chunks_ocd, chunks_null_class, y_ocd = preprocess_chunks_for_multiclass_test_one_handed(chunks, null_chunks, y)
labels = y_ocd.loc[:, "label"].squeeze()
assert len(labels) == len(chunks_ocd)
del chunks
del null_chunks
chunks_ocd_segmented, labels_ocd_segmented_multiclass, chunks_null_segmented, labels_null_segmented = segment_for_null_classification(chunks_ocd, chunks_null_class, window_size, labels)

assert len(set(labels)) == len(set(labels_ocd_segmented_multiclass))

# null class detector = ...

# multiclass clf without null
multi_class_df, labels_multi_class_classification = concat_chunks_for_feature_extraction(
        [chunks_ocd_segmented],
        [labels_ocd_segmented_multiclass])

X_multi_class_classification = extract_timeseries_features(multi_class_df, use_indoor=True, use_fingerprinting_approach=True, feature_set_config=feature_calculation_setting)
impute(X_multi_class_classification)
X_multi_class_classification_selected = select_features(X_multi_class_classification, labels_multi_class_classification)


# classifier = train multiclass clf
# test_data = pd.read_csv(...)
# test_segmented = segment_windows(...)
# for each in test_segmented:
#  classifier.predict(...)
