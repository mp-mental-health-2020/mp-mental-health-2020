from tsfresh import extract_relevant_features


def extract_timeseries_features(timeseries, y):
    features = extract_relevant_features(timeseries, y, column_id='action_id')
    return features
