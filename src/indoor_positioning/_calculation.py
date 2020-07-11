# Proximity

# get readings as df
# group by major minor
# batch (0.5, 0.75, 1) -> mean, median (mean rounded in direction of median?)
# filter -> Kalman (filtering before or after batching??)
import math

import numpy as np
import pandas as pd

import file_handling
from src.indoor_positioning import get_file_as_data_frame, get_recording_as_data_frame, get_specific_indoor_recording


def test_get_beacons_for_proximity_approach():
    recording = get_specific_indoor_recording()
    df = get_recording_as_data_frame(recording)

    # filter out incorrect placed beacons
    df = df[df["minor"] != 2]
    df = df[df["minor"] != 10]

    new_df = get_beacons_for_proximity_approach(df)
    print()


def get_beacons_for_proximity_approach(df, duration=1000, aggregation_function="mean"):
    grouped = df.groupby(["major", "minor"])
    series = df["timestamp"]
    start_timestamp = series.min()

    df = grouped.apply(batch_data, duration=duration, start_timestamp=start_timestamp, aggregation_function=aggregation_function)
    # TODO: sort by
    # merge batches with identical timestamp by using the maximum rssi value -> strongest signal

    # TODO: remove debugging stuff
    number_of_distinct_timestamps = df["timestamp"].unique()

    df2 = df.groupby(["timestamp"])["rssi"].max()
    df2 = df2.to_frame().reset_index()
    df = df.merge(df2, how="right", on=["timestamp", "rssi"])
    df.drop("major", axis=1, inplace=True)
    df.sort_values(by="timestamp", inplace=True)
    df.reset_index(inplace=True, drop=True)
    return df


def get_multiple_beacons_for_proximity_approach():
    recording = get_specific_indoor_recording()
    df = get_recording_as_data_frame(recording)

    # filter out incorrect placed beacons
    # df = df[df["minor"] != 2]
    # df = df[df["minor"] != 10]

    grouped = df.groupby(["major", "minor"])
    series = df["timestamp"]
    start_timestamp = series.min()
    df = grouped.apply(batch_data, duration=1000, start_timestamp=start_timestamp)

    timestamps = df["timestamp"].drop_duplicates()

    draw_threshold = 5
    new_df = pd.DataFrame(columns=df.columns)
    for timestamp in timestamps:
        selected_df = df[df["timestamp"] == timestamp]
        max_rssi = selected_df["rssi"].max()
        s = selected_df[selected_df["rssi"] > (max_rssi - draw_threshold)]
        new_df = new_df.append(s, ignore_index=True)

    labels = new_df["minor"].apply(get_label)
    new_df["labels"] = labels

    new_df["timestamp"] = new_df["timestamp"] - new_df["timestamp"].min()
    new_df.drop("major", axis=1, inplace=True)
    new_df.sort_values(by="timestamp", inplace=True)
    new_df.reset_index(inplace=True, drop=True)
    return new_df


def batch_data(df: pd.DataFrame, start_timestamp=None, duration=500, aggregation_function="mean"):
    # normalize timestamps over all beacons
    if not start_timestamp:
        start_timestamp = df["timestamp"].get_values()[0]

    # TODO: this assumes that major and minor are constant
    #  for methods like highest/lowest etc. this would not work
    major = df["major"].to_numpy()[0]
    minor = df["minor"].to_numpy()[0]

    # create indices for batching based on duration
    indices = ((df["timestamp"] - start_timestamp) / duration).apply(math.floor)
    frame = {'indices': indices, 'rssi': df["rssi"]}

    df_new = pd.DataFrame(frame)
    # apply aggregation function for batched data
    if not callable(aggregation_function):
        aggregation_function = get_aggregation_function(aggregation_function)
    df_new = df_new.groupby("indices").apply(aggregation_function)

    # calculate batched timestamps based on shared timestamp
    timestamps = df_new["indices"] * duration + start_timestamp

    df_new["timestamp"] = timestamps
    df_new["major"] = [major] * len(timestamps)
    df_new["minor"] = [minor] * len(timestamps)
    df_new.drop(columns=["indices"], inplace=True)
    return df_new


def apply_kalman_filter():
    # irrelevant for now
    pass


def get_aggregation_function(function_name):
    if function_name == "mean":
        return np.mean
    elif function_name == "median":
        return np.median


def get_label(minor):
    # Version 1 from 29.06.2020
    if minor == 1:
        return ["broken"]
    elif minor == 2:
        return ["hand-sanitizer"]
    elif minor == 3:
        return ["oven", "fridge", "sink"]
    elif minor == 4:
        return ["sink"]
    elif minor == 5:
        return ["window"]
    elif minor == 6:
        return ["window"]
    elif minor == 7:
        return []
    elif minor == 8:
        return []
    elif minor == 9:
        return ["window"]
    elif minor == 10:
        return []
    else:
        return ["unknown"]


def get_room(minor):
    # Version 1 from 29.06.2020
    if minor == 1:
        return "broken"
    elif minor == 2:
        return "entrance"
    elif minor == 3:
        return "kitchen"
    elif minor == 4:
        return "bathroom men"
    elif minor == 5:
        return "main room"
    elif minor == 6:
        return "main room"
    elif minor == 7:
        return "hallway"
    elif minor == 8:
        return "hallway"
    elif minor == 9:
        return "second room"
    elif minor == 10:
        return "bathroom women"
    else:
        return "unknown"


# Trilateration / Multilateration


# Fingerprinting


def test_duration():
    start = 1593784111687
    end = 1593785187858
    first = 1593784111733
    last = 1593785187852
    print(((end - start) / 1000) / 60)
    print(first - start)
    print(end - last)


def test_visualize_recordings():
    experiment_dir_path = "../../data/phyphox/full recordings/"
    experiment_dirs = file_handling.get_sub_directories(experiment_dir_path)
    for directory in experiment_dirs:
        # if "duration" not in directory:
        #    continue
        try:
            if "ariane" in directory.lower():
                indoor_file = file_handling.get_file_names_in_directory_for_pattern(directory, "*.json")[0]
                indoor_data_frame = get_file_as_data_frame(indoor_file)
                visualize_rssi_values(indoor_data_frame)
        except IndexError:
            # we don't have an indoor recording for this recording session
            pass


def test_visualize():
    recording = get_specific_indoor_recording()
    df = get_recording_as_data_frame(recording)
    visualize_rssi_values(df)


def visualize_rssi_values(df):
    import matplotlib.pyplot
    start_timestamp = df["timestamp"].get_values()[0]
    indices = df["timestamp"] - start_timestamp
    indices /= 1000
    indices /= 60
    matplotlib.pyplot.plot(indices, df["rssi"])
    matplotlib.pyplot.show()
