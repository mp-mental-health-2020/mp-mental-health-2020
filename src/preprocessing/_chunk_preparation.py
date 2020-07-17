import pandas as pd

from shared_constants import ACTION_ID_COL


def merge_left_and_right_chunk(chunk_left, chunk_right, action_id, chunk_indoor=None):
    """

    Parameters
    ----------
    chunk_left: dataframe containing chunk from left hand
    chunk_right: dataframe containing chunk from right hand
    chunk_indoor: dataframe containing chunk from indoor data
    action_id: id of the action (not the activity)
    Returns a merged dataframe with only one action_id column
    -------

    """
    # reset index first: this will set the indices to [0,1,2,3...] instead of timestamps which might not match between chunks
    # this ensures that we can join by index
    c_l = chunk_left.reset_index()
    c_r = chunk_right.reset_index(drop=True)

    c_i = None

    # TODO: make sure to sync correctly
    if chunk_indoor is not None:
        c_i = chunk_indoor.reset_index(drop=True)

    if ACTION_ID_COL in list(c_r.columns):
        c_r.drop(columns=[ACTION_ID_COL], inplace=True)
    # make sure that we have different column names for the data from right and left hand
    c_r.columns = [str(col) + '_right' for col in c_r.columns]
    # TODO: test that column count is correct
    # TODO: test that we only have one column with action_id
    # TODO: test that we have no NaNs

    if c_i is not None:
        c_all = pd.concat([c_l, c_r, c_i], axis=1)
    else:
        c_all = pd.concat([c_l, c_r], axis=1)

    c_all.set_index('index', inplace=True)
    c_all[ACTION_ID_COL] = action_id
    # TODO: test that the timestamp is the index again
    return c_all


def preprocess_chunks_for_null_test(chunks, null_chunks):
    """
    For the binary null classification append action id and merge the left and right chunk
    Parameters
    ----------
    chunks
    null_chunks

    Returns preprocessed chunks for ocd and null class
    -------

    """
    assert len(chunks["right"]) != 0
    assert len(null_chunks["right"]) != 0

    chunks_ocd = []
    chunks_length = len(chunks["right"])
    # append action id and merge left and right chunk
    for c_r, c_l, action_id in zip(chunks["right"], chunks["left"], range(chunks_length)):
        chunks_ocd.append(merge_left_and_right_chunk(c_l, c_r, action_id))

    chunks_null_class = []

    null_action_ids = range(len(chunks_ocd), len(chunks_ocd) + chunks_length)
    assert set(range(chunks_length)).isdisjoint(null_action_ids)

    for c_r, c_l, action_id in zip(null_chunks["right"], null_chunks["left"], null_action_ids):
        if len(c_l):
            c_both = merge_left_and_right_chunk(c_l, c_r, action_id)
            chunks_null_class.append(c_both)

    return chunks_ocd, chunks_null_class


def preprocess_chunks_for_null_test_with_indoor(chunks, null_chunks):
    """
    For the binary null classification append action id and merge the left and right chunk
    Parameters
    ----------
    chunks
    null_chunks

    Returns preprocessed chunks for ocd and null class
    -------

    """
    assert len(chunks["right"]) != 0
    assert len(null_chunks["right"]) != 0

    chunks_ocd = []
    chunks_length = len(chunks["right"])
    # append action id and merge left and right chunk
    for right_chunk, left_chunk, indoor_chunk, action_id in zip(chunks["right"], chunks["left"], chunks["indoor"], range(chunks_length)):
        chunks_ocd.append(merge_left_and_right_chunk(left_chunk, right_chunk, action_id, chunk_indoor=indoor_chunk))

    chunks_null_class = []

    null_action_ids = range(len(chunks_ocd), len(chunks_ocd) + chunks_length)
    assert set(range(chunks_length)).isdisjoint(null_action_ids)

    for right_chunk, left_chunk, indoor_chunk, action_id in zip(null_chunks["right"], null_chunks["left"], null_chunks["indoor"], null_action_ids):
        if len(left_chunk):
            chunks_null_class.append(merge_left_and_right_chunk(left_chunk, right_chunk, action_id, chunk_indoor=indoor_chunk))

    return chunks_ocd, chunks_null_class


def concat_chunks_for_feature_extraction(chunks, labels):
    """

    Parameters
    ----------
    chunks: list of lists of chunks (which can be segmented but don't need to be)
    labels: list of lists of the belonging labels

    Returns concatenated dataframe with all the recorded frames and the vector of labels belonging to the actions
    -------

    """

    assert len(chunks) == len(labels)
    assert len(chunks[0]) == len(labels[0])

    # flatten the chunks (will do something like chunks[0] + chunks[1] + ...
    chunks_flat_list = [c for c_list in chunks for c in c_list]
    concat_df = pd.concat(chunks_flat_list).reset_index(drop=True)
    concat_labels = pd.concat(labels)

    # the dataframe should have way more lines than the label vector as it contains all the recorded frames
    assert len(concat_df) > len(labels)
    return concat_df, concat_labels
