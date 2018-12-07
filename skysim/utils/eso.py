"""Utilities for working with the ESO Advanced Sky Model

See http://www.eso.org/sci/software/pipelines/skytools/skymodel
for details.
"""
import os.path
import tempfile
import hashlib
import json

import astropy.table


def call_skycalc(params):
    """Call the ESO Advanced Sky Calculator.

    Requires a network connection and that the skycalc_cli package
    is installed.

    The input parameters and output columns are documented at
    https://www.eso.org/observing/etc/doc/skycalc/helpskycalccli.html

    This function is relatively slow so the caching wrapper
    :func:`get_skycalc` should normally be used instead.

    Parameters
    ----------
    params : dict
        Dictionary of input parameters.

    Returns
    -------
    astropy.table.Table
        Table of output results.
    """
    try:
        import skycalc_cli.skycalc
    except ImportError as e:
        print('The skycalc_cli package is not installed.')
        return

    sc = skycalc_cli.skycalc.SkyModel()
    try:
        sc.callwith(params)
    except SystemExit:
        # Ignore a server error, which sometimes happens but
        # appears to be harmless.
        pass
    with tempfile.NamedTemporaryFile(mode='w+b', buffering=0) as tmpf:
        tmpf.write(sc.data)
        t = astropy.table.Table.read(tmpf.name)
    return t


_cache = {}

def get_skycalc(params, cache_path=None, verbose=False):
    """Caching frontend to call_skycalc.

    Uses a memory cache and an optional disk cache.

    Repeat calls with the same parameters are detected using an
    MD5 hash.

    Parameters
    ----------
    params : dict
        Dictionary of input parameters.
    cache_path : str or None
        Use this existing directory to cache results, or only use
        a memory cache when None.
    verbose : bool
        Print the comments from this skycalc run when True,
        using :func:`get_comments`.
    
    Returns
    -------
    astropy.table.Table
        Table of output results.
    """
    t = None
    # Create a hash of the requested parameters to use as a ~unique key.
    key = hashlib.md5(repr(params).encode('utf8')).hexdigest()
    # Is this request already cached in memory?
    global _cache
    if key in _cache:
        t = _cache[key]
    elif cache_path is not None:
        # Look for this file in the disk cache.
        if not os.path.isdir(cache_path):
            raise ValueError('No such path: {}.'.format(cache_path))
        name = os.path.join(cache_path, 'skycalc_{}.fits'.format(key))
        if os.path.exists(name):
            t = astropy.table.Table.read(name)
    if t is None:
        # If we get here, we need to call the network CLI.
        t = call_skycalc(params)
        # Cache this result in memory.
        _cache[key] = t
        if cache_path is not None:
            t.write(name, overwrite=False)
    if verbose:
        print(get_comments(t))
    return t


def get_comments(t):
    """Return comments in table metadata as single string.
    """
    return '\n'.join(t.meta['comments'])


def get_params(t):
    """Return parameter dictionary by parsing comments.

    Parameters
    ----------
    astropy.table.Table
        Table returned by :func:`get_skycalc` or :func:`call_skycalc`.

    Returns
    -------
    dict
        Dictionary of parameters parsed from the metadata comments
        using the standard library json parser.
    """
    comments = t.meta['comments']
    idx = comments.index('Input parameters:') + 1
    return json.loads('\n'.join(comments[idx:]))
