from sklearn.preprocessing import StandardScaler
from tsfresh import select_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd

from classification.classification import classify_all
from data_reading.phyphox import read_experiments_in_dir
from features import extract_timeseries_features
from file_handling import get_sub_directories
from preprocessing import concat_chunks_for_feature_extraction, preprocess_chunks_for_null_test, \
    preprocess_chunks_for_null_test_with_indoor, \
    segment_for_null_classification, segment_windows
from visualization._visualization import swarm_plot_top_features


def run_multiclass_classification(experiment_dir_path, experiment_dirs_selected, use_indoor, window_size, feature_calculation_setting):
    experiment_dirs = get_sub_directories(experiment_dir_path)
    experiment_dirs = [exp_dir for exp_dir in experiment_dirs if exp_dir.split("/")[-1] in experiment_dirs_selected]
    # Read data
    sample_rate = 50
    chunks, null_chunks, y = read_experiments_in_dir(experiment_dirs, sample_rate, drop_lin_acc=True, require_indoor=use_indoor)

    del experiment_dir_path
    del experiment_dirs
    print("Finished reading data")

    # we only need the y vector for the multi class clf
    y.reset_index(inplace=True)
    labels = y.loc[:, "label"].squeeze()

    # Preprocess data
    if use_indoor:
        chunks_ocd, chunks_null_class = preprocess_chunks_for_null_test_with_indoor(chunks, null_chunks)
    else:
        chunks_ocd, chunks_null_class = preprocess_chunks_for_null_test(chunks, null_chunks)

    del chunks
    del null_chunks
    # Segmentation

    chunks_ocd_segmented, labels_ocd_segmented, chunks_null_segmented, labels_null_segmented = segment_for_null_classification(chunks_ocd,
                                                                                                                               chunks_null_class,
                                                                                                                               window_size)

    assert len(labels_null_segmented) != 0
    labels_ocd_multiclass = labels.reset_index(drop=True)
    _, labels_ocd_segmented_multiclass = segment_windows(chunks_ocd, labels_ocd_multiclass.to_numpy(), window_size)
    del chunks_ocd
    del chunks_null_class

    assert len(set(labels_ocd_multiclass)) == len(set(labels_ocd_segmented_multiclass))

    # reuse chunks_ocd_segmented from the segmentation for the binary classifier
    assert len(labels_ocd_segmented_multiclass) == len(chunks_ocd_segmented)

    mulit_class_df, labels_multi_class_classification = concat_chunks_for_feature_extraction(
        [chunks_ocd_segmented, chunks_null_segmented],
        [labels_ocd_segmented_multiclass, labels_null_segmented])
    assert len(set(labels_multi_class_classification)) == len(set(labels_ocd_segmented_multiclass)) + 1

    labels_multi_class_classification = labels_multi_class_classification.str.replace("  ", " ").str.strip()

    assert set(labels_multi_class_classification) == {'checking oven',
                                                      'cleaning cup',
                                                      'cleaning floor',
                                                      'cleaning leg',
                                                      'cleaning table',
                                                      'cleaning window',
                                                      'drying hands',
                                                      'null class',
                                                      'pulling door',
                                                      'pulling hair',
                                                      'pushing door',
                                                      'sitting down',
                                                      'standing up',
                                                      'walking',
                                                      'washing hands'}

    # Feature extraction for multi class OCD activities incl null
    X_multi_class_classification = extract_timeseries_features(mulit_class_df, use_indoor=use_indoor,
                                                               feature_set_config=feature_calculation_setting)
    # Feature selection for multi class OCD activities incl null
    impute(X_multi_class_classification)
    X_multi_class_classification_selected = select_features(X_multi_class_classification,
                                                            labels_multi_class_classification)
    scaler = StandardScaler()
    X_multi_class_classification_scaled = scaler.fit_transform(X_multi_class_classification_selected)

    print("Multi class classification: using indoor: {}; FC params: {}; window_size {}".format(use_indoor,feature_calculation_setting.__class__.__name__, window_size))
    classify_all(X_multi_class_classification_scaled, labels_multi_class_classification)


    # TODO: store in file

def test_run_multiclass_recordings_clf():
    experiment_dir_path = "../../data/phyphox/full recordings/"
    # Ana-2, Ariane, Julian, Wiki
    experiment_dirs_selected = ["Ana-2", "Ariane", "Julian", "Wiktoria"]

    use_indoor = True
    window_size = 50
    # MinimalFCParameters, ComprehensiveFCParameters, EfficientFCParameters
    feature_calculation_setting = MinimalFCParameters()


    run_multiclass_classification(experiment_dir_path=experiment_dir_path,
                                  experiment_dirs_selected=experiment_dirs_selected,
                                  use_indoor=use_indoor,
                                  feature_calculation_setting=feature_calculation_setting,
                                  window_size=window_size)

experiment_dir_path = "../../data/phyphox/full recordings/"
# Ana-2, Ariane, Julian, Wiki
experiment_dirs_selected = ["Ana-2", "Ariane", "Julian", "Wiktoria"]
feature_calculation_setting_min = MinimalFCParameters()
feature_calculation_setting_efficient = EfficientFCParameters()

# min, indoor, 50
run_multiclass_classification(experiment_dir_path=experiment_dir_path,
                                  experiment_dirs_selected=experiment_dirs_selected,
                                  use_indoor=True,
                                  feature_calculation_setting=feature_calculation_setting_min,
                                  window_size=50)

# min, no indoor, 50
run_multiclass_classification(experiment_dir_path=experiment_dir_path,
                                  experiment_dirs_selected=experiment_dirs_selected,
                                  use_indoor=False,
                                  feature_calculation_setting=feature_calculation_setting_min,
                                  window_size=50)

# min, indoor, 100
run_multiclass_classification(experiment_dir_path=experiment_dir_path,
                                  experiment_dirs_selected=experiment_dirs_selected,
                                  use_indoor=True,
                                  feature_calculation_setting=feature_calculation_setting_min,
                                  window_size=100)

# min, no indoor, 100
run_multiclass_classification(experiment_dir_path=experiment_dir_path,
                                  experiment_dirs_selected=experiment_dirs_selected,
                                  use_indoor=False,
                                  feature_calculation_setting=feature_calculation_setting_min,
                                  window_size=100)

# efficient, indoor, 50
run_multiclass_classification(experiment_dir_path=experiment_dir_path,
                                  experiment_dirs_selected=experiment_dirs_selected,
                                  use_indoor=True,
                                  feature_calculation_setting=feature_calculation_setting_efficient,
                                  window_size=50)

# efficient, no indoor, 50
run_multiclass_classification(experiment_dir_path=experiment_dir_path,
                                  experiment_dirs_selected=experiment_dirs_selected,
                                  use_indoor=False,
                                  feature_calculation_setting=feature_calculation_setting_efficient,
                                  window_size=50)

# efficient, indoor, 100
run_multiclass_classification(experiment_dir_path=experiment_dir_path,
                                  experiment_dirs_selected=experiment_dirs_selected,
                                  use_indoor=True,
                                  feature_calculation_setting=feature_calculation_setting_efficient,
                                  window_size=100)

# efficient, no indoor, 100
run_multiclass_classification(experiment_dir_path=experiment_dir_path,
                                  experiment_dirs_selected=experiment_dirs_selected,
                                  use_indoor=False,
                                  feature_calculation_setting=feature_calculation_setting_efficient,
                                  window_size=100)
