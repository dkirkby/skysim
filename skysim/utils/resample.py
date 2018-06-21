"""Resampling utilities.
"""
import numpy as np

import scipy.interpolate


def centers_to_edges(centers, kind='cubic'):
    """Calculate bin edges from bin centers.

    Edges are calculated with interpolation (or extrapolation on the edges) from
    integer to half-integer indices.

    Parameters
    ----------
    centers : array
        1D array of N increasing center values with at least 2 values.
    kind : str or int
        Passed to :func:`scipy.interpolate.interp1d`. When N < 4,
        'linear' is always used.

    Returns
    -------
    array
        1D array of N+1 increasing bin edge values.
    """
    centers = np.asarray(centers)
    if len(centers.shape) != 1:
        raise ValueError('Expected 1D array of centers.')
    if len(centers) < 2:
        raise ValueError('Need at least 2 centers.')
    elif len(centers) < 4:
        kind= 'linear'
    if not np.all(np.diff(centers) > 0):
        raise ValueError('Expected increasing center values.')

    center_idx = np.arange(len(centers))
    interpolator = scipy.interpolate.interp1d(
        center_idx, centers, fill_value='extrapolate', copy=False,
        assume_sorted=True, kind=kind)

    edge_idx = np.arange(len(centers) + 1.) - 0.5
    return interpolator(edge_idx)
