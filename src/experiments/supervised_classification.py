from sklearn.preprocessing import StandardScaler
from tsfresh import select_features
from tsfresh.feature_extraction import MinimalFCParameters, EfficientFCParameters, ComprehensiveFCParameters
from tsfresh.utilities.dataframe_functions import impute
import pandas as pd
import os
import sys

from classification.classification import classify_all
from data_reading.phyphox import read_experiments_in_dir
from features import extract_timeseries_features
from file_handling import get_sub_directories
from preprocessing import concat_chunks_for_feature_extraction, preprocess_chunks_for_null_test, \
    preprocess_chunks_for_null_test_with_indoor, \
    segment_null_classification, segment_windows
from visualization._visualization import plot_duration_histogram, pca_2d, sne_2d
from output.output import output_figure

def run_multiclass_classification(experiment_dir_path, experiment_dirs_selected, use_indoor, window_size, feature_calculation_setting):
    path = os.getcwd()
    sub_folder = "indoor{}_features{}_windowSize{}/".format(use_indoor,feature_calculation_setting.__class__.__name__, window_size)
    path = path + "/output_experiments/multi/" + sub_folder
    if not os.path.exists(path):
        os.makedirs(path)
    sys.stdout = open(path + "console.txt", 'w')

    print("Multi class classification: using indoor: {}; FC params: {}; window_size {}".format(use_indoor,feature_calculation_setting.__class__.__name__, window_size))

    experiment_dirs = get_sub_directories(experiment_dir_path)
    experiment_dirs = [exp_dir for exp_dir in experiment_dirs if exp_dir.split("/")[-1] in experiment_dirs_selected]
    # Read data
    sample_rate = 50
    chunks, null_chunks, y = read_experiments_in_dir(experiment_dirs, sample_rate, drop_lin_acc=True, require_indoor=use_indoor)

    del experiment_dir_path
    del experiment_dirs
    print("Finished reading data")

    # we only need the y vector for the multi class clf
    #y.reset_index(inplace=True)
    labels = y.loc[:, "label"].squeeze()

    output_figure(fig=plot_duration_histogram(chunks["right"]), path=path, name="duration_histogram_activities", format="png")
    output_figure(fig=plot_duration_histogram(null_chunks["right"]), path=path, name="duration_histogram_null", format="png")

    # Preprocess data
    if use_indoor:
        chunks_ocd, chunks_null_class = preprocess_chunks_for_null_test_with_indoor(chunks, null_chunks)
    else:
        chunks_ocd, chunks_null_class = preprocess_chunks_for_null_test(chunks, null_chunks)

    del chunks
    del null_chunks
    # Segmentation

    chunks_ocd_segmented, labels_ocd_segmented, chunks_null_segmented, labels_null_segmented = segment_null_classification(chunks_ocd,
                                                                                                                           chunks_null_class,
                                                                                                                           window_size)

    assert len(labels_null_segmented) != 0

    labels_ocd_multiclass = labels.reset_index(drop=True)

    labels_ocd_multiclass = labels_ocd_multiclass.str.replace("  ", " ").str.strip()

    assert set(labels_ocd_multiclass) == {'checking oven',
                                          'cleaning cup',
                                          'cleaning floor',
                                          'cleaning leg',
                                          'cleaning table',
                                          'cleaning window',
                                          'drying hands',
                                          'pulling door',
                                          'pulling hair',
                                          'pushing door',
                                          'sitting down',
                                          'standing up',
                                          'walking',
                                          'washing hands'}

    _, labels_ocd_segmented_multiclass = segment_windows(chunks_ocd, labels_ocd_multiclass.to_numpy(), window_size)

    del chunks_ocd
    del chunks_null_class

    assert len(set(labels_ocd_multiclass)) == len(set(labels_ocd_segmented_multiclass))

    # reuse chunks_ocd_segmented from the segmentation for the binary classifier
    assert len(labels_ocd_segmented_multiclass) == len(chunks_ocd_segmented)

    multi_class_df, labels_multi_class_classification = concat_chunks_for_feature_extraction(
        [chunks_ocd_segmented, chunks_null_segmented],
        [labels_ocd_segmented_multiclass, labels_null_segmented])
    assert len(set(labels_multi_class_classification)) == len(set(labels_ocd_segmented_multiclass)) + 1

    # Feature extraction for multi class OCD activities incl null
    X_multi_class_classification = extract_timeseries_features(multi_class_df, use_indoor=use_indoor,
                                                               feature_set_config=feature_calculation_setting)
    # Feature selection for multi class OCD activities incl null
    impute(X_multi_class_classification)
    X_multi_class_classification_selected = select_features(X_multi_class_classification,
                                                            labels_multi_class_classification)
    scaler = StandardScaler()
    X_multi_class_classification_scaled = scaler.fit_transform(X_multi_class_classification_selected)

    output_figure(fig=pca_2d(X_multi_class_classification_scaled, labels_multi_class_classification,
           labels_multi_class_classification.unique(),
           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
            'C17', 'C18']), path=path, name="pca_2d_with_null", format="png")

    output_figure(fig=pca_2d(X_multi_class_classification_scaled, labels_multi_class_classification,
           labels_multi_class_classification.unique()[0:14],
           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
            'C17', 'C18']), path=path,name="pca_2d_without_null", format="png")

    output_figure(fig=sne_2d(X_multi_class_classification_scaled, labels_multi_class_classification,
           labels_multi_class_classification.unique(),
           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
            'C17', 'C18'], n_iter=1000, perplexity=30), path=path, name="sne_2d_with_null", format="png")

    output_figure(fig=sne_2d(X_multi_class_classification_scaled, labels_multi_class_classification,
           labels_multi_class_classification.unique()[0:14],
           ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16',
            'C17', 'C18'], n_iter=1000, perplexity=30), path=path,name="sne_2d_without_null", format="png")

    #print("Multi class classification: using indoor: {}; FC params: {}; window_size {}".format(use_indoor,feature_calculation_setting.__class__.__name__, window_size))
    classify_all(X_multi_class_classification_scaled, labels_multi_class_classification, path)

    # TODO: store in file

def run_experiments(config_file = './config_files/experiments_config.json'):
    import json
    with open(config_file) as f:
        config = json.load(f)
    classification_types = config["classification_types"]
    experiment_dir_paths = config["experiment_dir_paths"]
    experiment_dirs_selected = config["experiment_dirs_selected"]
    use_indoor = config["use_indoor"]
    feature_calculation_settings = config["feature_calculation_settings"]
    window_sizes = config["window_sizes"]
    exclude = config["exclude"]
    excluded_configuration = False

    for type in classification_types:
        for path in experiment_dir_paths:
            for experiment_dir in experiment_dirs_selected:
                for indoor in use_indoor:
                    for setting in feature_calculation_settings:
                        for size in window_sizes:
                            for i in range(len(exclude)):
                                if  not excluded_configuration and \
                                    type in exclude[i]["classification_types"] and \
                                    path in exclude[i]["experiment_dir_paths"] and \
                                    experiment_dir in exclude[i]["experiment_dirs_selected"] and \
                                    indoor in exclude[i]["use_indoor"] and \
                                    setting in exclude[i]["feature_calculation_settings"] and \
                                    size in exclude[i]["window_sizes"]:
                                        excluded_configuration = True
                            if not excluded_configuration:
                                if setting == "minimal": setting = MinimalFCParameters()
                                if setting == "efficient": setting = EfficientFCParameters()
                                if setting == "comprehensive": setting = ComprehensiveFCParameters()
                                if type == "multi":
                                    run_multiclass_classification(experiment_dir_path=path,
                                                              experiment_dirs_selected=experiment_dir,
                                                              use_indoor=indoor,
                                                              feature_calculation_setting=setting,
                                                              window_size=size)
                            #TODO implement binary classification
                            excluded_configuration = False

def test_run_multiclass_recordings_clf():
    run_experiments(config_file = './config_files/experiments_config.json')