from tsfresh import extract_features


def extract_timeseries_features(timeseries):
    features = extract_features(timeseries, column_id='action_id')
    return features
