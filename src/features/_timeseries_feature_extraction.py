import pandas as pd
from tsfresh import extract_features


def extract_timeseries_features(timeseries, use_indoor):
    if use_indoor:
        indoor_features = extract_indoor_feature(timeseries, column_id='action_id')
    features = extract_features(timeseries, column_id='action_id')
    if use_indoor:
        features.merge(indoor_features, right_index=True, left_index=True)
    return features


def extract_indoor_feature(data_frame, column_id="action_id"):
    indoor_df = data_frame.loc[:, ["action_id", "rssi", "minor"]]
    data_frame.pop("rssi")
    data_frame.pop("minor")
    indoor_df = indoor_df.groupby(column_id).apply(merge_indoor_values)
    return indoor_df


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
