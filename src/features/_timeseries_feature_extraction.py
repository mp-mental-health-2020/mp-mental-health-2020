import pandas as pd
import tsfresh
from tsfresh import extract_features


def extract_timeseries_features(timeseries, use_indoor, feature_set_config=tsfresh.feature_extraction.settings.ComprehensiveFCParameters()):
    if use_indoor:
        indoor_features = extract_indoor_feature(timeseries, column_id='action_id')
        timeseries = timeseries.drop(["rssi", "minor"], axis=1)
    assert "rssi" not in timeseries.columns
    assert "minor" not in timeseries.columns
    features = extract_features(timeseries, column_id='action_id', default_fc_parameters=feature_set_config)
    if use_indoor:
        features = features.merge(indoor_features, right_index=True, left_index=True)
        assert "minor" in features.columns
    return features


def extract_indoor_feature(data_frame, column_id="action_id"):
    indoor_df = data_frame.loc[:, ["action_id", "rssi", "minor"]]
    indoor_df.set_index("action_id", inplace=True)
    indoor_series = indoor_df.loc[:, "minor"]
    minors = indoor_series.groupby(level=0).apply(get_indoor_minor)
    return minors
    # work around to prevent indoor feature extraction from crashing:
    # return only the minor and drop the rssi
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
