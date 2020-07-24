import pandas as pd


def segment_windows(chunks, classes, window_size):
    """

    Parameters
    ----------
    chunks: timeseries chunks according to the action_ids: for testing this will take an array with only one item
    window_size: in the first place use 50, 100, 150 (=1,2,3s)

    Returns
    -------
    dataframe with new action ids (belonging to each segment)
    """

    new_chunks = []
    labels = []
    indices = []
    for c, l in zip(chunks, classes):
        for i in range(0, int(len(c) / window_size)):
            # TODO: test if samples shorter than window_size are removed
            c_new = c[i * window_size:(i + 1) * window_size]
            action_id = c["action_id"][0]
            # c_new["action_id"] = [(action_id, i)] * len(c_new)
            c_new["combined_id"] = [(action_id, i)] * len(c_new)
            c_new["action_id"] = [action_id] * len(c_new)
            c_new["segment_id"] = [i] * len(c_new)
            labels.append(l)
            indices.append((action_id, i))
            new_chunks.append(c_new)
    label_series = pd.Series(labels, index=indices)
    return new_chunks, label_series


def segment_null_classification(chunks_ocd, chunks_null_class, window_size):
    """

    Parameters
    ----------
    chunks_ocd
    chunks_null_class
    window_size

    Returns segmented chunks and labels for ocd and null class chunks
    -------

    """
    # new label for ocd activities
    labels_ocd_acts = pd.Series(["OCD activity"] * len(chunks_ocd))
    assert len(chunks_ocd[0].columns) == len(chunks_null_class[0].columns)
    chunks_ocd_segmented, labels_ocd_segmented = segment_windows(chunks_ocd, labels_ocd_acts.to_numpy(), window_size)

    null_labels = pd.Series(["null class"] * len(chunks_null_class))
    chunks_null_segmented, labels_null_segmented = segment_windows(chunks_null_class, null_labels.to_numpy(),
                                                                   window_size)

    # TODO: do we really need to return the labels here?
    return chunks_ocd_segmented, labels_ocd_segmented, chunks_null_segmented, labels_null_segmented
