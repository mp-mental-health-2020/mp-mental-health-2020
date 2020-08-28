import os
import sys
import warnings

import matplotlib
from sklearn.preprocessing import StandardScaler
from tsfresh import select_features
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from classification.classification import classify_all
from data_reading.phyphox import read_experiments_in_dir
from features import extract_timeseries_features
from file_handling import get_sub_directories
from preprocessing import concat_chunks_for_feature_extraction, \
    preprocess_chunks_for_null_test, \
    segment_for_null_classification, segment_windows
from output.output import output_figure
from shared_constants import SEGMENTATION_NO_OVERLAP, SEGMENTATION_OVERLAP
from visualization._visualization import pca_2d, plot_duration_histogram, sne_2d

def run_multiclass_classification(experiment_dir_path, experiment_dirs_selected, use_indoor, use_fingerprinting_approach, window_size,
                                  feature_calculation_setting, null_class_included, right_hand_only, segmentation_method, selected_activities=None):
    """
    Parameters
    ----------
    experiment_dir_path
    experiment_dirs_selected
    use_indoor : bool
        Use the indoor positioning as a feature.
    use_fingerprinting_approach : bool
        Use the fingerprinting instead of the most powerful signal approach.
    window_size
    feature_calculation_setting

    Returns
    -------

    """
    right_hand_only = False  # TODO rework
    path = os.getcwd()
    participants_folder = '-'.join(experiment_dirs_selected) + "/"
    selected_activities_str = ",".join(selected_activities).replace(" ", "") if selected_activities else ""
    sub_folder = "IL{}_fingerp{}_feat{}_winSize{}_segMeth{}_nullIncl{}/".format(use_indoor, use_fingerprinting_approach,
                                                                                                 feature_calculation_setting.__class__.__name__,
                                                                                                 window_size, segmentation_method, null_class_included)

    activities_sub_folder = selected_activities_str + "/"
    path = path + "/output_experiments/multi/" + participants_folder + sub_folder + activities_sub_folder
    if not os.path.exists(path):
        os.makedirs(path)
    #else:
    #    return
    sys.stdout = open(path + "console.txt", 'w')

    warnings.warn(participants_folder)
    warnings.warn(
        "Multi class classification: using indoor: {}; fingerprinting: {}; FC params: {}; window_size: {}; segmentation_method: {}; selected_activities: {}; null_class_included: {} \n\n".format(
            use_indoor, use_fingerprinting_approach, feature_calculation_setting.__class__.__name__, window_size, segmentation_method, selected_activities_str, null_class_included))
    print(participants_folder)
    print(
        "Multi class classification: using indoor: {}; fingerprinting: {}; FC params: {}; window_size: {}; segmentation_method: {}; selected_activities: {}; null_class_included: {}".format(use_indoor,
                                                                                                                                           use_fingerprinting_approach,
                                                                                                                                           feature_calculation_setting.__class__.__name__,
                                                                                                                                           window_size,
                                                                                                                                           segmentation_method,
                                                                                                                                           selected_activities_str,
                                                                                                                                           null_class_included))

    experiment_dirs = get_sub_directories(experiment_dir_path)
    experiment_dirs = [exp_dir for exp_dir in experiment_dirs if exp_dir.split("/")[-1] in experiment_dirs_selected]
    # Read data
    sample_rate = 50
    chunks, null_chunks, y = read_experiments_in_dir(experiment_dirs, sample_rate, drop_lin_acc=True, require_indoor=use_indoor,  selected_activities=selected_activities)

    # TODO test right hand only and change activities to only include both and right handed activities
    if right_hand_only:
        chunks = chunks["right"]
        null_chunks = null_chunks["right"]
    del experiment_dir_path
    del experiment_dirs
    print("Finished reading data")

    # we only need the y vector for the multi class clf
    y.reset_index(inplace=True)
    labels = y.loc[:, "label"].squeeze()

    output_figure(fig=plot_duration_histogram(chunks["right"]), path=path, name="duration_histogram_activities", format="png")
    output_figure(fig=plot_duration_histogram(null_chunks["right"]), path=path, name="duration_histogram_null", format="png")

    # Preprocess data
    chunks_ocd, chunks_null_class = preprocess_chunks_for_null_test(chunks, null_chunks, use_indoor)

    del chunks
    del null_chunks
    # Segmentation

    chunks_ocd_segmented, labels_ocd_segmented, chunks_null_segmented, labels_null_segmented = segment_for_null_classification(chunks_ocd = chunks_ocd,
                                                                                                                               chunks_null_class = chunks_null_class,
                                                                                                                               window_size = window_size,
                                                                                                                               segmentation_method = segmentation_method)

    assert len(labels_null_segmented) != 0

    labels_ocd_multiclass = labels.reset_index(drop=True)

    labels_ocd_multiclass = labels_ocd_multiclass.str.replace("  ", " ").str.strip()
    if selected_activities == None:
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
    else:
        assert set(labels_ocd_multiclass) == set(selected_activities)

    _, labels_ocd_segmented_multiclass = segment_windows(chunks = chunks_ocd, classes = labels_ocd_multiclass.to_numpy(), window_size = window_size, segmentation_method = segmentation_method)

    del chunks_ocd
    del chunks_null_class

    #assert len(set(labels_ocd_multiclass)) == len(set(labels_ocd_segmented_multiclass)) TODO adjust if activities are not in segments any more (200)

    # reuse chunks_ocd_segmented from the segmentation for the binary classifier
    assert len(labels_ocd_segmented_multiclass) == len(chunks_ocd_segmented)

    if null_class_included:
        multi_class_df, labels_multi_class_classification = concat_chunks_for_feature_extraction(
            [chunks_ocd_segmented, chunks_null_segmented],
            [labels_ocd_segmented_multiclass, labels_null_segmented])
        assert len(set(labels_multi_class_classification)) == len(set(labels_ocd_segmented_multiclass)) + 1
    else:
        multi_class_df, labels_multi_class_classification = concat_chunks_for_feature_extraction(
            [chunks_ocd_segmented],
            [labels_ocd_segmented_multiclass])
        assert len(set(labels_multi_class_classification)) == len(set(labels_ocd_segmented_multiclass))

    # Feature extraction for multi class OCD activities incl null
    X_multi_class_classification = extract_timeseries_features(multi_class_df, use_indoor=use_indoor,
                                                               use_fingerprinting_approach=use_fingerprinting_approach,
                                                               feature_set_config=feature_calculation_setting)

    # Feature selection for multi class OCD activities incl null
    impute(X_multi_class_classification)
    X_multi_class_classification_selected = select_features(X_multi_class_classification,
                                                            labels_multi_class_classification)

    scaler = StandardScaler()
    X_multi_class_classification_scaled = scaler.fit_transform(X_multi_class_classification_selected)

    color_hex_codes = ['#013896', '#e53242', '#00234d', '#de89a1', '#2e8b57', '#b7b7b7', '#e5c8ce', '#4b5c09', '#eca628', '#df6722', '#e6d26d', '#727a9a', '#c73b0f', '#00ffff', '#80c1ff', '#930dd4', '#438880', '#000000']

    output_figure(fig=pca_2d(X_multi_class_classification_scaled, labels_multi_class_classification,
                             labels_multi_class_classification.unique(),
                             color_hex_codes), path=path, name="pca_2d_with_null", format="png")
    if null_class_included:
        output_figure(fig=pca_2d(X_multi_class_classification_scaled, labels_multi_class_classification,
                                 labels_multi_class_classification.unique()[0:14],
                                 color_hex_codes), path=path, name="pca_2d_without_null", format="png")

    output_figure(fig=sne_2d(X_multi_class_classification_scaled, labels_multi_class_classification,
                             labels_multi_class_classification.unique(),
                             color_hex_codes, n_iter=1000, perplexity=30), path=path, name="sne_2d_with_null", format="png")
    if null_class_included:
        output_figure(fig=sne_2d(X_multi_class_classification_scaled, labels_multi_class_classification,
                                 labels_multi_class_classification.unique()[0:14],
                                 color_hex_codes, n_iter=1000, perplexity=30), path=path, name="sne_2d_without_null", format="png")

    classify_all(X_multi_class_classification_scaled, labels_multi_class_classification, path, binary=False)


def run_binary_classification(experiment_dir_path, experiment_dirs_selected, use_indoor, use_fingerprinting_approach, window_size,
                              feature_calculation_setting, segmentation_method, selected_activities=None):
    right_hand_only = False  # TODO rework
    path = os.getcwd()
    participants_folder = '-'.join(experiment_dirs_selected) + "/"
    selected_activities_str = ",".join(selected_activities).replace(" ", "") if selected_activities else ""
    sub_folder = "IL{}_fingerp{}_feat{}_winSize{}_segMeth{}/".format(use_indoor, use_fingerprinting_approach,
                                                                             feature_calculation_setting.__class__.__name__,
                                                                             window_size, segmentation_method)
    activities_sub_folder = selected_activities_str + "/"
    path = path + "/output_experiments/binary/" + participants_folder + sub_folder + activities_sub_folder
    if not os.path.exists(path):
        os.makedirs(path)
    #else:
    #    return
    sys.stdout = open(path + "console.txt", 'w')

    warnings.warn(participants_folder)
    warnings.warn("Binary classification: using indoor: {}; using fingerprinting: {}; FC params: {}; window_size: {}; segmentation_method: {}; selected_activities: {} \n\n".format(
        use_indoor, use_fingerprinting_approach, feature_calculation_setting.__class__.__name__, window_size, segmentation_method, selected_activities_str))
    print(participants_folder)
    print("Binary classification: using indoor: {}; using fingerprinting: {}; FC params: {}; window_size: {}; segmentation_method: {}; selected_activities: {}".format(
        use_indoor, use_fingerprinting_approach, feature_calculation_setting.__class__.__name__, window_size, segmentation_method, selected_activities_str))

    experiment_dirs = get_sub_directories(experiment_dir_path)
    experiment_dirs = [exp_dir for exp_dir in experiment_dirs if exp_dir.split("/")[-1] in experiment_dirs_selected]
    # Read data
    sample_rate = 50
    chunks, null_chunks, y = read_experiments_in_dir(experiment_dirs, sample_rate, drop_lin_acc=True, require_indoor=use_indoor, selected_activities=selected_activities)

    # TODO test right hand only and change activities to only include both and right handed activities
    if right_hand_only:
        chunks = chunks["right"]
        null_chunks = null_chunks["right"]
    del experiment_dir_path
    del experiment_dirs
    print("Finished reading data")

    # we only need the y vector for the multi class clf
    # y.reset_index(inplace=True)
    labels = y.loc[:, "label"].squeeze()

    output_figure(fig=plot_duration_histogram(chunks["right"]), path=path, name="duration_histogram_activities",
                  format="png")
    output_figure(fig=plot_duration_histogram(null_chunks["right"]), path=path, name="duration_histogram_null",
                  format="png")

    # Preprocess data
    chunks_ocd, chunks_null_class = preprocess_chunks_for_null_test(chunks, null_chunks, use_indoor)

    del chunks
    del null_chunks
    # Segmentation

    chunks_ocd_segmented, labels_ocd_segmented, chunks_null_segmented, labels_null_segmented = segment_for_null_classification(
        chunks_ocd = chunks_ocd,
        chunks_null_class = chunks_null_class,
        window_size = window_size,
        segmentation_method = segmentation_method)

    assert len(labels_null_segmented) != 0

    null_classification_df, labels_null_classification = concat_chunks_for_feature_extraction(
        [chunks_ocd_segmented, chunks_null_segmented],
        [labels_ocd_segmented, labels_null_segmented])
    assert len(set(labels_null_classification)) == 2
    X_null_class_classification = extract_timeseries_features(null_classification_df, use_indoor=use_indoor,
                                                              feature_set_config=feature_calculation_setting,
                                                              use_fingerprinting_approach=use_fingerprinting_approach)
    impute(X_null_class_classification)

    X_null_classification_selected = select_features(X_null_class_classification, labels_null_classification)

    scaler = StandardScaler()
    X_null_classification = scaler.fit_transform(X_null_classification_selected)

    output_figure(fig=pca_2d(X_null_classification, labels_null_classification,
                             labels_null_classification.unique(),
                             ['C1', 'C2']), path=path, name="pca_2d", format="png")

    output_figure(fig=sne_2d(X_null_classification, labels_null_classification,
                             labels_null_classification.unique(),
                             ['C1', 'C2'], n_iter=1000, perplexity=30), path=path, name="sne_2d", format="png")

    classify_all(X_null_classification, labels_null_classification, path, binary=True)

    # labels_null_classification.reset_index(drop=True)

    # add the old labels to the column names of the features again
    # X_null_classification_selected = pd.DataFrame(X_null_classification, columns=X_null_classification_selected.columns)

    # reduce the amount of selected features and append the labels as an extra column
    # X_y = pd.concat([X_null_classification_selected.iloc[:, :5], labels_null_classification.reset_index(drop=True)],
    #                axis=1)

    # label_vals = {11: "null class", 12: "OCD activity"}
    # rename the last column
    # cols = [c for c in X_y.columns]
    # cols[-1] = "class"
    # X_y.columns = cols
    # X_y.replace({"class": label_vals}, inplace=True)

    # output_figure(fig=swarm_plot_top_features(pd.DataFrame(X_y).reset_index()), path=path, name="swarm_plot_top_features", format="png")

def calculate_number_of_experiments(use_indoor, classification_types, experiment_dir_paths, experiment_dirs_selected, use_fingerprinting_approach, feature_calculation_settings, window_sizes, null_class_included, overlaps):
    # TODO fix calculation (binary - null class; selected_activities)
    if True in use_indoor:
        total_number_of_experiments_without_exclude = len(classification_types) * len(experiment_dir_paths) * len(
            experiment_dirs_selected) * (
                                                              len(use_indoor) + len(use_fingerprinting_approach) - 1) * len(
            feature_calculation_settings) * len(window_sizes) * len(
            null_class_included) * len(overlaps)
    else:
        total_number_of_experiments_without_exclude = len(classification_types) * len(experiment_dir_paths) * len(
            experiment_dirs_selected) * len(
            use_indoor) * len(feature_calculation_settings) * len(window_sizes) * len(null_class_included) * len(overlaps)
    return total_number_of_experiments_without_exclude



# TODO test
def run_experiments(config_file='./config_files/experiments_config.json'):
    import json
    with open(config_file) as f:
        config = json.load(f)
    classification_types = config["classification_types"]
    experiment_dir_paths = config["experiment_dir_paths"]
    experiment_dirs_selected = config["experiment_dirs_selected"]
    use_indoor = config["use_indoor"]
    use_fingerprinting_approach = config["use_fingerprinting_approach"]
    feature_calculation_settings = config["feature_calculation_settings"]
    window_sizes = config["window_sizes"]
    overlaps = config["overlaps"]
    selected_activities = config["activities"]
    null_class_included = config["null_class_included"]
    right_hand_only = [False]
    total_number_of_experiments_without_exclude = calculate_number_of_experiments(use_indoor, classification_types, experiment_dir_paths, experiment_dirs_selected, use_fingerprinting_approach, feature_calculation_settings, window_sizes, null_class_included, overlaps)
    number_of_current_experiment = 1
    exclude = config["exclude"]
    excluded_configuration = False

    for setting in feature_calculation_settings:
        for type in classification_types:
            orig_type = type
            for path in experiment_dir_paths:
                for experiment_dir in experiment_dirs_selected:
                    for indoor in use_indoor:
                        for fingerprinting in use_fingerprinting_approach:
                            if (not indoor) and fingerprinting: continue
                            for size in window_sizes:
                                if not selected_activities:
                                    selected_activities = [None]
                                for activities in selected_activities:
                                    if not activities:
                                        activities = None
                                    for included in null_class_included:
                                        type = orig_type
                                        if (selected_activities != [None]) and (activities is not None) and \
                                                (len(activities) == 1): type = "binary"
                                        if type == "binary" and (not included):
                                            continue
                                        if (activities is not None) and (len(activities) == 2) and (not included): type = "binary_from_multi"
                                        for right_hand in right_hand_only:
                                            for overlap in overlaps:
                                                for i in range(len(exclude)):
                                                    if not excluded_configuration and \
                                                            ((type in exclude[i]["classification_types"]) or
                                                             (type == "binary_from_multi" and "multi" in exclude[i]["classification_types"])) and \
                                                            path in exclude[i]["experiment_dir_paths"] and \
                                                            experiment_dir in exclude[i]["experiment_dirs_selected"] and \
                                                            indoor in exclude[i]["use_indoor"] and \
                                                            fingerprinting in exclude[i]["use_fingerprinting_approach"] and \
                                                            setting in exclude[i]["feature_calculation_settings"] and \
                                                            size in exclude[i]["window_sizes"] and \
                                                            included in exclude[i]["null_class_included"] and \
                                                            right_hand in exclude[i]["right_hand_only"]:
                                                        excluded_configuration = True
                                                if not excluded_configuration:
                                                    minimal = MinimalFCParameters()
                                                    del minimal['length']
                                                    efficient = EfficientFCParameters()
                                                    del efficient['length']
                                                    comprehensive = ComprehensiveFCParameters()
                                                    del comprehensive['length']
                                                    if setting == "minimal":
                                                        setting = minimal
                                                    if setting == "efficient":
                                                        setting = efficient
                                                    if setting == "comprehensive":
                                                        setting = comprehensive
                                                    if overlap:
                                                        segmentation_method = SEGMENTATION_OVERLAP
                                                    if not overlap:
                                                        segmentation_method = SEGMENTATION_NO_OVERLAP
                                                    warnings.warn("Execute experiment number: " + str(number_of_current_experiment) + "/" + str(
                                                        total_number_of_experiments_without_exclude))
                                                    if type == "multi":
                                                        run_multiclass_classification(experiment_dir_path=path,
                                                                                      experiment_dirs_selected=experiment_dir,
                                                                                      use_indoor=indoor,
                                                                                      use_fingerprinting_approach=fingerprinting,
                                                                                      feature_calculation_setting=setting,
                                                                                      window_size=size,
                                                                                      null_class_included=included,
                                                                                      right_hand_only=right_hand,
                                                                                      selected_activities=activities,
                                                                                      segmentation_method=segmentation_method)
                                                        matplotlib.pyplot.close("all")
                                                    if (type == "binary") or (type == "binary_from_multi"):
                                                        run_binary_classification(experiment_dir_path=path,
                                                                                  experiment_dirs_selected=experiment_dir,
                                                                                  use_indoor=indoor,
                                                                                  use_fingerprinting_approach=fingerprinting,
                                                                                  feature_calculation_setting=setting,
                                                                                  window_size=size,
                                                                                  selected_activities=activities,
                                                                                  segmentation_method=segmentation_method)
                                                        matplotlib.pyplot.close("all")
                                                    number_of_current_experiment += 1
                                                excluded_configuration = False


def test_run_multiclass_recordings_clf():
    run_experiments(config_file='/tmp/pycharm_project_688/src/experiments/config_files/experiments_config.json')
    #run_experiments(config_file='./src/experiments/config_files/experiments_config.json')


"""
def test_run_multiclass_classification():
    # for debugging purposes
    run_multiclass_classification(experiment_dir_path="../../data/phyphox/full recordings/", experiment_dirs_selected=["Wiktoria"],
                                  use_indoor=True,
                                  use_fingerprinting_approach=False, window_size=50, feature_calculation_setting=MinimalFCParameters())
"""