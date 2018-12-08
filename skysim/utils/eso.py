"""Utilities for working with the ESO Advanced Sky Model

See http://www.eso.org/sci/software/pipelines/skytools/skymodel
for details.
"""
import os.path
import tempfile
import hashlib
import json

import numpy as np

import astropy.table
import astropy.constants
import astropy.units as u

import skysim.zodiacal


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


def black_body_radiance(wlen, T, photon_units=True):
    """Calculate the black body radiance in energy or photon units.
    """
    hc = astropy.constants.h * astropy.constants.c
    # Calculate spectral radiance per wavelength unit in energy units.
    arg = hc / (wlen * astropy.constants.k_B * T)
    rad = 2 * hc * astropy.constants.c / wlen ** 5 / (np.exp(arg) - 1) / u.steradian
    if photon_units:
        # Convert to radiance in photons.
        energy_per_photon = (hc / wlen) / u.ph
        rad = rad / energy_per_photon
    return rad


def prepare_params(location, obstime, RA, DEC, wmin=300, wmax=1100, wdelta=1):
    """Prepare SkyCalc parameters for a single observation.

    Parameters
    ----------
    location : astropy.coordinates.EarthLocation
        Earth location of the observation.
    obstime : astropy.time.time
        Time of the observation.
    RA : float
        Right ascension of the telescope pointing in degrees.
    DEC : float
        Declination of the telescope pointing in degrees.

    Returns
    -------
    dict
        Dictionary of parameters suitable for calling :func:`get_skycalc`.
    """
    # Initialize the pointing, assuming an extragalactic target.
    pointing = astropy.coordinates.SkyCoord(
        ra=RA * u.deg, dec=DEC * u.deg, distance=1 * u.Gpc, frame='icrs')
    # Calculate moon and sun positions.
    sun = astropy.coordinates.get_sun(obstime)
    moon = astropy.coordinates.get_moon(obstime)
    moon_sun = sun.separation(moon)
    moon_tgt = pointing.separation(moon)
    # Calculate observer altitude angles, without refraction corrections.
    observer = astropy.coordinates.AltAz(
        location=location, obstime=obstime, pressure=0)
    obs_altaz = pointing.transform_to(observer)
    moon_altaz = moon.transform_to(observer)
    airmass = skysim.zodiacal.airmass_zodi(90 - obs_altaz.alt.to(u.deg).value)
    # Calculate ecliptic coordinates.
    ecliptic = astropy.coordinates.HeliocentricTrueEcliptic(obstime=obstime)
    ecl_lonlat = pointing.transform_to(ecliptic)
    # Ecliptic longitude is measured relative to the sun RA in the range 0-180.
    ecl_lon = np.fmod(np.abs((ecl_lonlat.lon - sun.ra).to(u.deg).value), 360)
    wrap = np.floor(ecl_lon / 180)
    ecl_lon = wrap * (360 - ecl_lon) + (1 - wrap) * ecl_lon
    # Pick the predefined observatory with the closest elevation.
    elev = location.height.to(u.m).value
    closest = np.argmin(
        [abs(elev - obs[1]) for obs in skysim.utils.eso.observatories])
    # Build the parameter dictionary.
    params = dict(
        airmass=airmass,
        moon_sun_sep=moon_sun.to(u.deg).value,
        moon_target_sep=moon_tgt.to(u.deg).value,
        moon_alt=moon_altaz.alt.to(u.deg).value,
        moon_earth_dist=1.0,
        ecl_lon=ecl_lon,
        ecl_lat=ecl_lonlat.lat.to(u.deg).value,
        season=0,
        time=0,
        vacair='vac',
        wmin=wmin,
        wmax=wmax,
        wdelta=wdelta,
        # The CLI identifies observatories by height, not by name.
        observatory=str(observatories[closest][1]),
    )
    return params

# Dictionary of predefined model observatories and their elevations in meters.
observatories = (
    ('VLT Cerro Paranal', 2640),
    ('La Silla', 2400),
    ('3060m', 3060),
)

params = {
    # Used for Table 5 of Noll 2012.
    'Noll2012 Default': dict(
        airmass=1.0,
        moon_sun_sep=0.0,
        moon_target_sep=180.0,
        moon_alt=-90.0,
        moon_earth_dist=1.0,
        ecl_lon=135.0,
        ecl_lat=90.0,
        msolflux=130.0,
        season=0,
        time=0,
        vacair='vac',
        wmin=300.0,
        wmax=4200.0,
        wdelta=5.0,
        observatory='2640',
    ),
    # Used for Figures 1, 6, 13, 17 of Noll 2012.
    # We do not include the thermal emission parameters since
    # these are not specified in Table 1 (although used in Fig. 1).
    'Noll2012 Demo Run': dict(
        airmass=1.003667671798522, # skysim.zodiacal.airmass_zodi(90 - 85.1)
        moon_sun_sep=77.9,
        moon_target_sep=51.3,
        moon_alt=41.3,
        moon_earth_dist=1.0,
        ecl_lon=-124.5,
        ecl_lat=-31.6,
        msolflux=205.5,
        season=4,
        time=3,
        vacair='air',
        wmin=300.0,
        wmax=4200.0,
        wdelta=5.0,
        observatory='2640',
    ),
}
