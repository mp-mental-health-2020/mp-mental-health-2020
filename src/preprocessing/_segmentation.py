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
    for c,l in zip(chunks,classes):
        for i in range(0, int(len(c) / window_size)):
            # TODO: test if samples shorter than window_size are removed
            c_new = c[i * window_size:(i + 1) * window_size]
            action_id = c["action_id"][0]
            c_new["action_id"] = [(action_id, i)] * len(c_new)
            labels.append(l)
            indices.append((action_id, i))
            new_chunks.append(c_new)
    label_series = pd.Series(labels, index=indices)
    return new_chunks, label_series
