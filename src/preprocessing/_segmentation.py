

def segment_windows(chunks, window_size):
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
    for c in chunks:
        for i in range(0, int(len(c) / window_size)):
            c_new = c[i * window_size:(i + 1) * window_size]
            #c_new.reset_index(inplace=True)
            action_id = c["action_id"][0]
            c_new["action_id"] = [(action_id, i)] * len(c_new)
            new_chunks.append(c_new)
    return new_chunks
