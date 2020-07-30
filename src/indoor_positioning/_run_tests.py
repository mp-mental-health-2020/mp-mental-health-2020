from file_handling import get_file_names_in_directory_for_pattern, get_sub_directories
from indoor_positioning import (get_file_as_data_frame, get_prepared_beacon_data, get_recording_as_data_frame, get_specific_indoor_recording)
from indoor_positioning._visualization import visualize_beacon_data_to_find_offset


def test_get_beacons_for_fingerprinting_approach():
    recording = get_specific_indoor_recording()
    df = get_recording_as_data_frame(recording)
    df_2 = get_prepared_beacon_data(df)
    print()


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
