import math

import numpy as np
import pandas as pd

from src.indoor_positioning import get_recording_as_data_frame, get_specific_indoor_recording

BEACON_MINORS = [2, 3, 4, 5, 6, 7, 8, 10]
MISSING_VALUE = -120


def test_get_beacons_for_proximity_approach():
    recording = get_specific_indoor_recording()
    df = get_recording_as_data_frame(recording)
    df_2 = get_beacons_for_fingerprinting_approach(df)
    print()


def get_beacons_for_fingerprinting_approach(df):
    df = prepare_data_frame(df)

    series = df["timestamp"]
    start_timestamp = series.min()
    end_timestamp = series.max()

    result = pd.DataFrame({"timestamp": []})

    for minor in BEACON_MINORS:
        current_df = df[df["minor"] == minor]
        current_df.drop(["major", "minor"], axis=1, inplace=True)
        current_df = batch_data(current_df, start_timestamp=start_timestamp, end_timestamp=end_timestamp, duration=500)
        current_df.rename(columns={"rssi": minor}, inplace=True)
        result = result.merge(current_df, on="timestamp", how="outer")

    return result


def prepare_data_frame(df):
    recorded_beacons = df["minor"].unique()
    if 11 in recorded_beacons:
        # special setup used for the recordings of charlotte and lisa
        # filter out 5 and 6 and map 11 to 5 and 9 to 6
        df = df[df["minor"] != 5]
        df = df[df["minor"] != 6]
        df["minor"].replace(11, 5, inplace=True)
        df["minor"].replace(9, 6, inplace=True)
    else:
        # filter out configuration beacon
        df = df[df["minor"] != 9]
    return df


def batch_data(df: pd.DataFrame, start_timestamp=None, end_timestamp=None, duration=500, aggregation_function="mean"):
    # normalize timestamps over all beacons
    if not start_timestamp:
        start_timestamp = df["timestamp"].get_values()[0]

    # create indices for batching based on duration
    timestamps = (((df["timestamp"] - start_timestamp) / duration).apply(math.floor)) * duration + start_timestamp
    frame = {'timestamp': timestamps, 'rssi': df["rssi"]}

    df_new = pd.DataFrame(frame)
    # apply aggregation function for batched data
    if not callable(aggregation_function):
        aggregation_function = get_aggregation_function(aggregation_function)
    df_new = df_new.groupby("timestamp").apply(aggregation_function)
    df_new.reset_index(drop=True, inplace=True)

    # add missing timestamps
    number_of_runs = math.ceil((end_timestamp - start_timestamp) / duration)
    all_timestamps = np.array([start_timestamp + duration * index for index in range(number_of_runs)])

    if not df_new.empty:
        current_timestamps = df_new["timestamp"].values
    else:
        current_timestamps = []

    missing_timestamps = [timestamp for timestamp in all_timestamps if timestamp not in current_timestamps]
    missing_values = [MISSING_VALUE] * len(missing_timestamps)

    missing_frame = {'timestamp': missing_timestamps, 'rssi': missing_values}
    missing_df = pd.DataFrame(missing_frame)

    result = df_new.append(missing_df, ignore_index=True)
    return result.sort_values("timestamp", axis=0, ascending=True)


def get_aggregation_function(function_name):
    if function_name == "mean":
        return np.mean
    elif function_name == "median":
        return np.median

# interpolate beacon data based for each minor
# batched -> if no value is available -> set to -100 / -120
# have one column for each minor
# TODO:
#  - in feature calculation we now create the fingerprint for the given window
#       - one based on the rssi values
#       - one based on ranking (and maybe even weighted ranking)
#  - see if the bug created by tuple as index is still a problem -> switch to multi ?
