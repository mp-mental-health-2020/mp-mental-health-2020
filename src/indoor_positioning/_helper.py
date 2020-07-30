import os
import random

import pandas as pd

from src.file_handling import get_file_names_in_directory_for_pattern, get_project_directory, read_json_file

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
