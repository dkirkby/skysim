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


def resample_binned(edges_out, edges_in, hist_in, zero_pad=True):
    """Flux conserving resampler of binned data.

    Parameters
    ----------
    edges_out : array
        1D array of M >= 2 output bin edges, in increasing order.
    edges_in : array
        1D array of N >= 2 input bin edges, in increasing order.
    hist_in : array
        1D array of N-1 input bin values.
    zero_pad : bool
        When True, allow the output edges to extend beyond the input
        edges and assume that the input histogram is zero outside of
        its extent.  When False, raises a ValueError if extrapolation
        would be required.

    Returns
    -------
    array
        1D array of M-1 resampled bin values.
    """
    if len(edges_out) < 2:
        raise ValueError('Need at least one output bin.')
    if len(edges_in) < 2:
        raise ValueError('Need at least one input bin.')
    if len(hist_in) != len(edges_in) - 1:
        raise ValueError('Unexpected length of hist_in.')
    binsize_out = np.diff(edges_out)
    if np.any(binsize_out <= 0):
        raise ValueError('Expecting increasing edges_out.')
    binsize_in = np.diff(edges_in)
    if np.any(binsize_in <= 0):
        raise ValueError('Expecting increasing edges_in.')
    if not zero_pad and ((edges_out[0] < edges_in[0]) or (edges_out[-1] > edges_in[-1])):
        raise ValueError('Ouput bins extend beyond input bins but zero_pad is False.')
    if (edges_out[0] >= edges_in[-1]) or (edges_out[-1] <= edges_in[0]):
        raise ValueError('Input and output bins do not overlap.')
    # Align output edges to input edges.
    idx = np.searchsorted(edges_in, edges_out)
    # Loop over output bins.
    nin = len(edges_in) - 1
    nout = len(edges_out) - 1
    hist_out = np.zeros(nout)
    hi = idx[0]
    for i in range(nout):
        lo = hi
        hi = idx[i + 1]
        if (lo > nin) or (hi == 0):
            # This bin does not overlap the input.
            continue
        if lo == hi:
            # Output bin is fully embedded within an input bin: give it a linear share.
            hist_out[i] = binsize_out[i] / binsize_in[lo - 1] * hist_in[lo - 1]
            continue
        # Calculate fraction of first input bin overlapping this output bin.
        if lo > 0:
            hist_out[i] += (edges_in[lo] - edges_out[i]) / binsize_in[lo - 1] * hist_in[lo - 1]
        # Calculate fraction of last input bin overlaping this output bin.
        if hi <= nin:
            hist_out[i] += (edges_out[i + 1] - edges_in[hi - 1]) / binsize_in[hi - 1] * hist_in[hi - 1]
        # Add input bins fully contained within this output bin.
        if hi > lo + 1:
            hist_out[i] += np.sum(hist_in[lo:hi - 1])
    return hist_out


def resample_density(x_out, x_in, y_in, zero_pad=True):
    """Flux conserving resampling of density samples.

    By "density" we mean that the integral of y(x) is the conserved flux.

    This function is just a wrapper around :func:`resample_binned` that:
     - Estimates input bin edges.
     - Multiplies each density y(x[i]) by the bin width to obtain bin contents.
     - Resamples the binned data.
     - Divides output bin values by output bin widths to obtain densities.
    
    The special case of a single output sample location is handled with linear
    interpolation of the input densities, so is not really flux conserving but
    probably what you want in this case.

    Parameters
    ----------
    x_out : array
        1D array of M >= 1 output sample locations.
    x_in : array
        1D array of N >= 2 input sample locations.
    y_in : array
        1D array of N input sample densities.
    zero_pad : bool
        When True, allow the output edges to extend beyond the input
        edges and assume that the input histogram is zero outside of
        its extent.  When False, raises a ValueError if extrapolation
        would be required.

    Returns
    -------
    array
        1D array of M output densities.
    """
    x_out = np.asarray(x_out)
    if len(x_out.shape) == 0 or len(x_out) == 1:
        # Resampling to a single value.
        if not zero_pad and ((x_out < np.min(x_in)) or (x_out > np.max(x_in))):
            raise ValueError('Cannot resample to point outside support when zero_pad is False.')
        # Linearly interpolate x_out in (x_in, y_in).
        return np.interp(x_out, x_in, y_in, left=0., right=0.)
    edges_out = centers_to_edges(x_out)
    edges_in = centers_to_edges(x_in)
    hist_in = y_in * np.diff(edges_in)
    hist_out = resample_binned(edges_out, edges_in, hist_in, zero_pad)
    return hist_out / np.diff(edges_out)