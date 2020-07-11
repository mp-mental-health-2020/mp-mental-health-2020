import pandas as pd

from shared_constants import ACTION_ID_COL


def merge_left_and_right_chunk(chunk_left, chunk_right, action_id, chunk_indoor=None):
    """

    Parameters
    ----------
    chunk_left: dataframe containing chunk from left hand
    chunk_right: dataframe containing chunk from left hand
    chunk_right: dataframe containing chunk from indoor data
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
