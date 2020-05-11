import matrixprofile as mp
import numpy as np


def compute_and_visualize_motifs(timeseries, windows, k=3):
    """
    computes the best k motifs for the given windows
    Parameters
    ----------
    timeseries
    windows
    k

    Returns found motifs
    -------

    """

    profile = mp.compute(timeseries, n_jobs=-1, windows=windows)
    profile_motifs = mp.discover.motifs(profile, k=k)
    mp.visualize(profile_motifs)
    return profile_motifs


def motif_count_mean(profile_motifs):
    """
    The amount of found patterns per result is the number of motifs + the number of neighbors
    This method calculates the mean of the amount of found patterns across all motif results.
    Parameters
    ----------
    profile_motifs: dict containing information about the discovered motifs

    Returns mean of the amount of found patterns across all motif results
    -------

    """
    motifs = profile_motifs['motifs']
    mean_motif_count = np.array([len(m['motifs']) + len(m['neighbors']) for m in motifs]).mean()
    return mean_motif_count
