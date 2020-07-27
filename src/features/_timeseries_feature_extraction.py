import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters

from indoor_positioning.constant import BEACON_MINORS


def extract_timeseries_features(timeseries, use_indoor, use_fingerprinting_approach, feature_set_config=None):
    if not feature_set_config:
        feature_set_config = ComprehensiveFCParameters()
    if use_indoor:
        indoor_df = timeseries.loc[:, BEACON_MINORS + ["action_id", "segment_id"]]
        indoor_features = indoor_df.groupby(["action_id", "segment_id"]).apply(np.mean)
        indoor_features.drop(["action_id", "segment_id"], axis=1, inplace=True)

        if not use_fingerprinting_approach:
            # use only strongest beacon -> base for maybe using labels later on
            indoor_features = pd.DataFrame(indoor_features.idxmax(axis=1), columns=["minor"])

    timeseries.drop(["action_id", "segment_id"] + BEACON_MINORS, axis=1, inplace=True)
    features = extract_features(timeseries, column_id='combined_id', default_fc_parameters=feature_set_config)

    if use_indoor:
        # index may be altered but order and count remained the same
        indoor_features.set_index(features.index, inplace=True)
        features = features.merge(indoor_features, right_index=True, left_index=True)
    return features
