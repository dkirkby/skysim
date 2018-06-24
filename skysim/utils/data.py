"""Utilities for loading and caching package data files.
"""
import numpy as np

import astropy.table
import astropy.utils.data


_cache = {'atmosphere': None, 'solarspec': None}


def get(name, force_load=False):
    """Return the named table.

    Load the table from this package's data directory, if necessary,
    or return the previously loaded table.

    The supported names are:
    * `solarspec`: see :func:`scripts.prepare_data.solarspec` for details.
    * `atmosphere`: see :func:`scripts.prepare_data.atmosphere` for details.

    Parameters
    ----------
    force_load : bool
        Force the table to be loaded from disk when True.

    Returns
    -------
    astropy.table.Table
        Requested table.
    """
    if name not in _cache:
        raise ValueError(f'Invalid name: "{name}".')
    cached = _cache.get(name, None)
    if force_load or cached is None:
        path = astropy.utils.data._find_pkg_data_path(f'../data/{name}.fits')
        cached = astropy.table.Table.read(path)
        _cache[name] = cached
    return cached
