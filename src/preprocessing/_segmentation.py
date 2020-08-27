import pandas as pd

from shared_constants import SEGMENTATION_NO_OVERLAP


def segment_windows(chunks, classes, window_size, segmentation_method=SEGMENTATION_NO_OVERLAP):
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
        # for the 'overlap' method the only difference is that the number of segments changes
        number_of_segments = int(len(c) / window_size) if segmentation_method==SEGMENTATION_NO_OVERLAP else int(len(c) / (window_size*0.5))-1
        for i in range(0, number_of_segments):
            # TODO: test if samples shorter than window_size are removed
            if segmentation_method==SEGMENTATION_NO_OVERLAP:
                c_new = c[i * window_size:(i + 1) * window_size]
            else:
                c_new = c[((i * window_size)-int(window_size*0.5*i)):(((i + 1) * window_size)-int(window_size*0.5*i))]
            action_id = c["action_id"][0]
            c_new["combined_id"] = [(action_id, i)] * len(c_new)
            c_new["action_id"] = [action_id] * len(c_new)
            c_new["segment_id"] = [i] * len(c_new)
            labels.append(l)
            indices.append((action_id, i))
            new_chunks.append(c_new)
    label_series = pd.Series(labels, index=indices)
    return new_chunks, label_series


def segment_for_null_classification(chunks_ocd, chunks_null_class, window_size, labels_ocd_acts=None, segmentation_method=SEGMENTATION_NO_OVERLAP):
    """

    Parameters
    ----------
    chunks_ocd
    chunks_null_class
    window_size
    segmentation_method: different segmentation methods available are
        SEGMENTATION_NO_OVERLAP: cut off the last segment of the action if it doesn't have the length of the window size
        SEGMENTATION_OVERLAP: use a 50% overlap within the action samples for segmentation
    labels_ocd_acts: can be passed for multiclass classification

    Returns segmented chunks and labels for ocd and null class chunks
    -------

    """
    # new label for ocd activities
    if labels_ocd_acts is None:
        labels_ocd_acts = pd.Series(["OCD activity"] * len(chunks_ocd))
    assert len(chunks_ocd[0].columns) == len(chunks_null_class[0].columns)
    assert len(labels_ocd_acts) == len(chunks_ocd)
    chunks_ocd_segmented, labels_ocd_segmented = segment_windows(chunks_ocd, labels_ocd_acts.to_numpy(), window_size, segmentation_method)

    null_labels = pd.Series(["null class"] * len(chunks_null_class))
    chunks_null_segmented, labels_null_segmented = segment_windows(chunks_null_class, null_labels.to_numpy(),
                                                                   window_size, segmentation_method)

    # TODO: do we really need to return the labels here?
    return chunks_ocd_segmented, labels_ocd_segmented, chunks_null_segmented, labels_null_segmented
