import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters


def extract_timeseries_features(timeseries, use_indoor, use_fingerprinting_approach, feature_set_config=None):
    if not feature_set_config:
        feature_set_config = ComprehensiveFCParameters()
    if use_indoor:
        if use_fingerprinting_approach:
            indoor_df = timeseries.loc[:, [2, 3, 4, 5, 6, 7, 8, 10, "action_id", "segment_id"]]
            indoor_features = indoor_df.groupby(["action_id", "segment_id"]).apply(np.mean)
            indoor_features.drop(["action_id", "segment_id"], axis=1, inplace=True)
            timeseries.drop([2, 3, 4, 5, 6, 7, 8, 10, ], axis=1, inplace=True)
            # TODO: re-create tuple index for merge
        else:
            indoor_features = extract_indoor_feature(timeseries, column_id=['action_id', "segment_id"])
            timeseries.drop(["rssi", "minor"], axis=1, inplace=True)
    timeseries.drop(["action_id", "segment_id"], axis=1, inplace=True)
    features = extract_features(timeseries, column_id='combined_id', default_fc_parameters=feature_set_config)
    if use_indoor:
        # One is tuple, one is multi index
        # features.reset_index(drop=True, inplace=True)
        # indoor_feature.reset_index(drop=True, inplace=True)
        indoor_features.set_index(features.index, inplace=True)
        features = features.merge(indoor_features, right_index=True, left_index=True)
    return features


def extract_indoor_feature(data_frame, column_id=None):
    if not column_id:
        column_id = ["action_id", "segment_id"]
    indoor_df = data_frame.loc[:, ["action_id", "segment_id", "rssi", "minor"]]

    # indoor_series = indoor_df.loc[:, "minor"]
    # minors = indoor_series.groupby(column_id).apply(get_indoor_minor)
    # work around to prevent indoor feature extraction from crashing:
    # return only the minor and drop the rssi
    # return minors

    indoor_df = indoor_df.groupby(column_id).apply(merge_indoor_values)
    return indoor_df


def get_indoor_minor(grouped_data):
    counts = grouped_data.value_counts()
    most_frequent_minor = counts.sort_values(ascending=False).index[0]
    return most_frequent_minor


def merge_indoor_values(grouped_data):
    # find most consistent beacon
    counts = grouped_data["minor"].value_counts()
    most_frequent_minor = counts.sort_values(ascending=False).index[0]

    # aggregate beacon signal strength values
    filter_minor = grouped_data[grouped_data["minor"] == most_frequent_minor]
    highest_rssi = filter_minor["rssi"].max()
    try:
        result = pd.DataFrame(data=[[most_frequent_minor, highest_rssi]], columns=["minor", "rssi"])
    except KeyError:
        # Bug: one entry is missing action_id -> not sure why for now
        return pd.DataFrame([], columns=["minor", "rssi"])
    return result
