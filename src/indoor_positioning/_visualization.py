import matplotlib.pyplot

import file_handling
from indoor_positioning import get_file_as_data_frame, get_recording_as_data_frame, get_specific_indoor_recording


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


def visualize_rssi_values(df):
    start_timestamp = df["timestamp"].get_values()[0]
    indices = df["timestamp"] - start_timestamp
    indices /= 1000
    indices /= 60
    matplotlib.pyplot.plot(indices, df["rssi"])
    matplotlib.pyplot.show()


def test_visualize_recordings():
    experiment_dir_path = "../../data/phyphox/full recordings/"
    experiment_dirs = file_handling.get_sub_directories(experiment_dir_path)
    for directory in experiment_dirs:
        # if "duration" not in directory:
        #    continue
        try:
            if "julius" in directory.lower():
                indoor_file = file_handling.get_file_names_in_directory_for_pattern(directory, "*.json")[0]
                print(indoor_file)
                indoor_data_frame = get_file_as_data_frame(indoor_file)
                visualize_rssi_values(indoor_data_frame)
        except IndexError:
            # we don't have an indoor recording for this recording session
            pass


def test_visualize():
    recording = get_specific_indoor_recording()
    df = get_recording_as_data_frame(recording)
    visualize_rssi_values(df)


def test_print_duration():
    start = 1594988156329
    first_data = 1594988156449
    end = 1594994625236
    last_data = 1594989956376
    duration = end - start
    data_duration = last_data - first_data
    print(duration / 1000 / 60)
    print(data_duration / 1000 / 60)
    print(first_data - start)
    print(end - last_data)
