import pandas as pd

import file_handling
from data_reading.phyphox import read_experiment
from file_handling import get_sub_directories
from indoor_positioning import get_beacons_for_proximity_approach, get_file_as_data_frame
from preprocessing import align_data, merge_left_and_right_chunk, segment_windows, set_time_delta_as_index


def test_supervised_classification():
    experiment_dir_path = "../../data/phyphox/full recordings/"
    experiment_dirs = get_sub_directories(experiment_dir_path)
    # complete_experiments_indices = [1,2,3,5,7]
    # experiment_dirs = [experiment_dirs[i] for i in complete_experiments_indices]
    sample_rate = 50
    action_data_frames = {"right": [], "left": [], "indoor": []}
    null_data_frames = {"right": [], "left": [], "indoor": []}
    y_columns = ["start", "end", "label", "hand"]

    y = get_data_and_labels(experiment_dirs, y_columns, pd.DataFrame(columns=y_columns), action_data_frames, null_data_frames, sample_rate)

    # append the activity label (as int) and the action id to the dataframe
    # we need to do this to be able to extract time series features later

    labels = y["label"].unique()
    label_ids = {label: index for label, index in zip(labels, range(0, len(labels)))}

    # list of tuples (left chunk, right chunk)
    chunks_two_handed = []

    y = y.replace(label_ids)

    # TODO: maybe reset index
    for index, action_row in y.iterrows():
        action_id = index
        two_handed_chunk = []
        for hand, current_chunk_data_list in action_data_frames.items():

            # TODO: handle indoor here
            if hand == "indoor":
                continue

            current_data_chunk = current_chunk_data_list[index]
            two_handed_chunk.append(current_data_chunk)
            one_handed_chunk = current_data_chunk
            one_handed_chunk["action_id"] = action_id

        two_handed_chunk = merge_left_and_right_chunk(two_handed_chunk[0], two_handed_chunk[1], action_id)
        chunks_two_handed.append(two_handed_chunk)
        del two_handed_chunk
        del action_id

    labels = y.loc[:, "label"].squeeze()
    window_size = 100

    # prepare null chunks
    null_class_chunks = []

    # TODO: assert that this list is disjoint to the list of action ids from activities
    null_action_ids = range(len(chunks_two_handed), len(chunks_two_handed) + len(null_data_frames["right"]))
    for c_r, c_l, action_id in zip(null_data_frames["right"], null_data_frames["left"], null_action_ids):
        if len(c_l):
            c_both = merge_left_and_right_chunk(c_l, c_r, action_id)
            null_class_chunks.append(c_both)

    # new label id for ocd activities
    labels_ocd_acts = pd.Series([labels.max() + 2] * len(chunks_two_handed))
    chunks_ocd_activities, labels_ocd_acts = segment_windows(chunks_two_handed, labels_ocd_acts.to_numpy(), window_size)

    # TODO: add indoor to segment_windows
    null_labels = pd.Series([labels.max() + 1] * len(null_class_chunks))
    null_class_chunks, null_labels = segment_windows(null_class_chunks, null_labels.to_numpy(), window_size)

    null_classification_concat = pd.concat(chunks_ocd_activities + null_class_chunks).reset_index(drop=True)
    # features_two_handed_null_test = extract_timeseries_features(null_classification_concat)


def get_data_and_labels(experiment_dirs, y_columns, y: pd.DataFrame, action_data_frames, null_data_frames, sample_rate):
    for directory in experiment_dirs:
        offsets = {}

        try:
            with open(directory + "/offset.txt") as f:
                for line in f:
                    (key, val) = line.split(": ")
                    offsets[key.lower()] = val
        except FileNotFoundError:
            continue

        data_frames = read_experiment(directory, offsets=offsets)
        data_frames = {key: align_data(data_frame, listening_rate=1000 / sample_rate, reference_sensor=None) for key, data_frame in
                       data_frames.items()}

        try:
            indoor_file = file_handling.get_file_names_in_directory_for_pattern(directory, "*.json")[0]
            indoor_data_frame = get_file_as_data_frame(indoor_file)

            # filter out incorrect placed beacons
            indoor_data_frame = indoor_data_frame[indoor_data_frame["minor"] != 2]
            indoor_data_frame = indoor_data_frame[indoor_data_frame["minor"] != 10]

            new_df = get_beacons_for_proximity_approach(indoor_data_frame)
            indoor_data_frame = new_df
            indoor_data_frame = set_time_delta_as_index(indoor_data_frame, origin_timestamp_unit='ms',
                                                        output_timestamp_unit="milliseconds",
                                                        timestamp_key="timestamp")
            indoor_data_frame.sort_index(inplace=True)
            # TODO: filter out minor 2 and 10 for now
            # TODO: align needs to be done on aggregated data
            # TODO: do we really need alignment -> for now yes
            data_frames["indoor"] = align_data(indoor_data_frame, interpolation_method="previous", listening_rate=1000 / sample_rate,
                                               reference_sensor=None)
            del indoor_data_frame
            del new_df
            del indoor_file
        except IndexError:
            # we don't have an indoor recording for this recording session
            continue

        # get relevant data from action annotations
        y_actions = pd.read_csv(directory + "/annotations.tsv", delimiter="\t", header=None)
        y_actions = y_actions.iloc[:, [3, 5, 8]]

        # get relevant data from hand annotations
        y_hands = pd.read_csv(directory + "/hands.tsv", delimiter="\t", header=None)
        y_hands = y_hands.iloc[:, [8]]

        # combine labels
        y_current = pd.concat([y_actions, y_hands], axis=1)
        y_current.columns = y_columns

        y = y.append(y_current)

        # iterate over the annotations and split the timeseries in chunks
        for key, df in data_frames.items():
            if key in action_data_frames:
                action_data_frames[key] += [df.iloc[int(annotation["start"] * sample_rate):int(annotation["end"] * sample_rate)] for i, annotation in
                                            y_current.iterrows()]
                # null chunks are everything in between annotations
                null_data_frames[key] += [df.iloc[int(annotation["end"] * sample_rate):int(y_current.iloc[i + 1:i + 2]["start"] * sample_rate)] for
                                          i, annotation in y_current.iterrows() if i < len(y_current) - 1]

        break
    return y