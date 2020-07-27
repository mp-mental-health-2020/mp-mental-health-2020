import math

import numpy as np
import pandas as pd

from indoor_positioning.constant import BEACON_MINORS, MISSING_VALUE


def get_prepared_beacon_data(df, beacon_minors=None, mirrored_setup=False, missing_value=MISSING_VALUE):
    df = prepare_data_frame(df, mirrored_setup=mirrored_setup)

    series = df["timestamp"]
    start_timestamp = series.min()
    end_timestamp = series.max()

    result = pd.DataFrame({"timestamp": []})

    if not beacon_minors:
        beacon_minors = BEACON_MINORS

    for minor in beacon_minors:
        current_df = df[df["minor"] == minor]
        current_df.drop(["major", "minor"], axis=1, inplace=True)
        current_df = batch_data(current_df, start_timestamp=start_timestamp, end_timestamp=end_timestamp, duration=500, missing_value=missing_value)
        current_df.rename(columns={"rssi": minor}, inplace=True)
        result = result.merge(current_df, on="timestamp", how="outer")

    return result


def prepare_data_frame(df, mirrored_setup=False):
    if mirrored_setup:
        # Special setup used for the recordings of Charlotte and Lisa. They sat in room 2 with beacons 11 and 9 simulating beacon 5 and 6 as to not
        # mess with original setup. Therefore filter out 5 and 6 and map 11 to 5 and 9 to 6.
        df = df[df["minor"] != 5]
        df = df[df["minor"] != 6]
        df["minor"].replace(11, 5, inplace=True)
        df["minor"].replace(9, 6, inplace=True)
    else:
        # filter out configuration beacon
        df = df[df["minor"] != 9]
    return df


def batch_data(df: pd.DataFrame, start_timestamp=None, end_timestamp=None, duration=500, aggregation_function="mean", missing_value=MISSING_VALUE):
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
    missing_values = [missing_value] * len(missing_timestamps)

    missing_frame = {'timestamp': missing_timestamps, 'rssi': missing_values}
    missing_df = pd.DataFrame(missing_frame)

    result = df_new.append(missing_df, ignore_index=True)
    return result.sort_values("timestamp", axis=0, ascending=True)


def get_aggregation_function(function_name):
    if function_name == "mean":
        return np.mean
    elif function_name == "median":
        return np.median
