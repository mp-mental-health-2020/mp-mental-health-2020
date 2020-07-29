import os
import random
import warnings

import pandas as pd

from src.file_handling import get_file_names_in_directory_for_pattern, get_project_directory, get_sub_directories, read_json_file

# Example json:
# {
#   "advertisingPacketMap": {},
#   "endTimestamp": 1590398862483,
#   "startTimestamp": 1590398852481
# }

INDOOR_POSITIONING_DIRECTORY_NAME = "indoor_positioning"
COLUMNS = [
    "timestamp",
    "rssi",
    "major",
    "minor"
]


# receive timestamps for periods in respective rooms
# create major and minor mapping for place and room


def get_indoor_data_directory():
    """
    It is assumed that the data directory is at the same level as the git project. If this is not the case you need to adjust the path here. The files
    are then further ordered by their respective recording app.

    Returns
    -------
        Directory containing the phyphox data
    """
    return os.path.join(get_project_directory(), "data", INDOOR_POSITIONING_DIRECTORY_NAME)


def filter_files(file_names):
    # TODO: filter for timestamps -> only thing available in name
    #  could filter more accurately by reading the file (major, minor, etc.)
    pass


def get_random_indoor_recording():
    file_names = get_file_names_in_directory_for_pattern(get_indoor_data_directory(), "*.json")
    random_file = random.sample(file_names, 1)[0]
    return read_json_file(random_file)


def get_specific_indoor_recording():
    file_names = get_file_names_in_directory_for_pattern(get_indoor_data_directory(), "*.json")
    for file_name in file_names:
        if "_-2_" in file_name:
            random_file = file_name
    return read_json_file(random_file)


def get_file_as_data_frame(file_path):
    recording = read_json_file(file_path)
    packets = recording["advertisingPacketList"]
    return pd.DataFrame(packets, columns=COLUMNS)


def get_recording_as_data_frame(recording):
    packets = recording["advertisingPacketList"]
    return pd.DataFrame(packets, columns=COLUMNS)


def test_reading():
    recording = get_random_indoor_recording()
    packets = recording["advertisingPacketList"]
    df = pd.DataFrame(packets, columns=COLUMNS)
    print()


def map_major(major):
    if major == 1:
        return "DHC Ground Floor"
    else:
        warnings.warn("Location for major {} is unknown. Please add it to the mapping.".format(major))
        return "Unknown Location"


def visualize_beacon_data_to_find_offset(df, minor=9, directory=None):
    import matplotlib.pyplot
    # get data for specific beacon
    start_timestamp = df["timestamp"].get_values()[0]
    df = df[df["minor"] == minor]
    df = df[df["rssi"] >= -55]
    df["timestamp"] -= start_timestamp
    df["timestamp"] /= 1000

    if not df.empty:
        indices = df["timestamp"]
        # indices /= 1000
        # indices /= 60
        matplotlib.pyplot.plot(indices, df["rssi"])
        # matplotlib.pyplot.show()


def test_get_offset_for_indoor():
    experiment_dirs = get_sub_directories("../../data/phyphox/full recordings/")
    for directory in experiment_dirs:
        if any([user in directory for user in ["Hung", "Julian", "Ana-2", "Wiktoria", "Ariane"]]):
            files = get_file_names_in_directory_for_pattern(directory, "*.json")
            if not files:
                continue
            print(directory)
            indoor_file = get_file_names_in_directory_for_pattern(directory, "*.json")[0]
            indoor_data_frame = get_file_as_data_frame(indoor_file)
            visualize_beacon_data_to_find_offset(indoor_data_frame, directory=directory)
