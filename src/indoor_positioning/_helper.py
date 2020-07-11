import os
import random
import warnings

import pandas as pd

from src import file_handling
from src.file_handling import get_project_directory

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
    file_names = file_handling.get_file_names_in_directory_for_pattern(get_indoor_data_directory(), "*.json")
    random_file = random.sample(file_names, 1)[0]
    return file_handling.read_json_file(random_file)


def get_specific_indoor_recording():
    file_names = file_handling.get_file_names_in_directory_for_pattern(get_indoor_data_directory(), "*.json")
    for file_name in file_names:
        if "_-2_" in file_name:
            random_file = file_name
    return file_handling.read_json_file(random_file)


def get_file_as_data_frame(file_path):
    recording = file_handling.read_json_file(file_path)
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
