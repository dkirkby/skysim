"""Calculate metadata relevant to sky simulations.
"""
import warnings

import numpy as np

import astropy.coordinates
import astropy.table
import astropy.constants
import astropy.units as u

import skysim.utils.data


def get_sky_metadata(location, obstime, pointing, pressure=None,
                     airtemp=None, ephemeris='builtin'):
    """Calculate sky metadata for specified time and pointing.

    Any dates outside the tabulated solarflux data will have their
    ``solar_flux`` values set to zero.

    The ``obstime`` and ``pointing`` inputs must have compatible
    shapes for broadcasting.

    Parameters
    ----------
    location : astropy.coordinates.earth.EarthLocation
        The observatory earth location.
    obstime : astropy.time.Time
        One or more observing times when metadata should be calculated.
    pointing : astropy.coordinates.SkyCoord
        The telescope pointing at each input time. If the coordinates are
        purely angular, with no distance specified, targets are assumed to
        be extragalactic for the purposes of calculating ecliptic coordinates.
    pressure : astropy.units.Quantity or None
        Atmospheric pressure to use for refraction corrections. When None, use
        :func:`nominal_pressure`. Set to zero (units optional) for no
        refraction corrections.
    airtemp : astropy.units.Quantity or None
        Air temperature to use for refraction corrections. Ignored when
        ``pressure`` is zero.
    ephemeris : str
        Ephemeris to use for the moon.

    Returns
    -------
    astropy.table.Table
        Table of computed sky metadata for each (time, pointing) observed from
        the specified location.
    """
    try:
        out = np.broadcast(obstime, pointing)
    except ValueError:
        raise ValueError('obstime and pointing have incompatible shapes.')
    output = astropy.table.Table()
    # Calculate apparent local sidereal time in degrees.
    lst = np.atleast_1d(obstime.sidereal_time(
        'apparent', longitude=location.lon).to(u.deg).value)
    output['lst'] = astropy.table.Column(
        np.broadcast_to(lst, out.shape),
        unit='degree', description='Apparent local sidereal time')
    # Get the positions of the sun and moon at each time.
    sun = astropy.coordinates.get_sun(obstime)
    moon = astropy.coordinates.get_moon(obstime, ephemeris=ephemeris)
    # Calculate the moon phase angle (degrees) and illuminated fraction
    # (following astroplan.moon)
    elongation = sun.separation(moon)
    moon_phase_angle = np.arctan2(
        sun.distance*np.sin(elongation),
        moon.distance - sun.distance*np.cos(elongation)
        ).to(u.deg).value
    output['moon_illum_frac'] = astropy.table.Column(
        1 + np.cos(np.deg2rad(moon_phase_angle)) / 2.0,
        description='Illuminated fraction (0=new, 1=full)')
    # Define the local observing frame.
    if pressure is None:
        pressure = nominal_pressure(location.height, airtemp)
    observer = astropy.coordinates.AltAz(
        location=location, obstime=obstime,
        pressure=pressure, temperature=airtemp)
    # Calculate the pointing, sun and moon local angles in degrees.
    obs_altaz = pointing.transform_to(observer)
    output['obs_alt'] = astropy.table.Column(
        obs_altaz.alt.to(u.deg).value, unit='degree',
        description='Observation altitude angle')
    output['obs_az'] = astropy.table.Column(
        obs_altaz.az.to(u.deg).value, unit='degree',
        description='Observation azimuth angle')
    sun_altaz = sun.transform_to(observer)
    output['sun_alt'] = astropy.table.Column(
        sun_altaz.alt.to(u.deg).value, unit='degree',
        description='Sun altitude angle')
    output['sun_az'] = astropy.table.Column(
        sun_altaz.az.to(u.deg).value, unit='degree',
        description='Sun azimuth angle')
    moon_altaz = moon.transform_to(observer)
    output['moon_alt'] = astropy.table.Column(
        moon_altaz.alt.to(u.deg).value, unit='degree',
        description='Moon altitude angle')
    output['moon_az'] = astropy.table.Column(
        moon_altaz.az.to(u.deg).value, unit='degree',
        description='Moon azimuth angle')
    # Calculate the sun and moon separation angles in degrees.
    output['sun_sep'] = astropy.table.Column(
        sun.separation(pointing).to(u.deg).value, unit='degree',
        description='Sun-pointing opening angle')
    output['moon_sep'] = astropy.table.Column(
        moon.separation(pointing).to(u.deg).value, unit='degree',
        description='Moon-pointing opening angle')
    # Calculate the heliocentric ecliptic (lon, lat) in degrees.
    if not u.m.is_equivalent(pointing.cartesian.x.unit):
        # Silently assume that targets are extragalactic (1Gpc) for the
        # purposes of calculating their ecliptic coordinates.
        pointing = astropy.coordinates.SkyCoord(
            ra=pointing.ra, dec=pointing.dec, distance=1 * u.Gpc,
            frame=pointing.frame)
    ecliptic = astropy.coordinates.HeliocentricTrueEcliptic(obstime=obstime)
    ecliptic_lonlat = pointing.transform_to(ecliptic)
    # Longitude is measured relative to the sun RA.
    output['ecl_lon'] = astropy.table.Column(
        (ecliptic_lonlat.lon - sun.ra).to(u.deg).value, unit='degree',
        description='Heliocentric ecliptic longitude ' +
                    'of pointing relative to sun')
    output['ecl_lat'] = astropy.table.Column(
        ecliptic_lonlat.lat.to(u.deg).value, unit='degree',
        description='Heliocentric ecliptic latitude of pointing')
    # Lookup monthly average 10.7cm solar flux for each observation.
    t_solarflux = skysim.utils.data.get('solarflux')
    month_number = np.array([
        12 * (D.year - 1986) + (D.month - 1)
        for D in obstime.reshape(-1).datetime]).reshape(obstime.shape)
    solarflux = np.zeros(obstime.shape, np.float32)
    valid = (month_number >= 0) & (month_number < len(t_solarflux))
    if np.any(~valid):
        warnings.warn('Some observations are outside tabulated solar flux.')
    solarflux[valid] = t_solarflux[month_number[valid]]['OBSFLUX']
    output['solar_flux'] = astropy.table.Column(
        solarflux, unit=t_solarflux['OBSFLUX'].unit,
        description='Monthly average observed solar flux at 10.7cm')

    return output


def nominal_pressure(elevation, airtemp=None):
    """Calculate the nominal pressure in a standard atmosphere.

    Uses a model with a linear decrease in temperature with
    elevation from sea level. For details, see
    https://en.wikipedia.org/wiki/Vertical_pressure_variation

    Parameters
    ----------
    elevation : astropy.units.Quantity
        Elevation above sea level to use.
    airtemp : astropy.units.Quantity or None
        Air temperature at the specified elevation to use for
        the standard atmosphere.  When None, assume 15C
        at sea level with a constant lapse rate.

    Returns
    -------
    astropy.units.Quantity
        Pressure at the specified elevation in a standard atmosphere.
    """
    # The atmosphere constant in astropy < 2.0 was renamed to atm in 2.0.
    try:
        P0 = astropy.constants.atm
    except AttributeError:
        # Fallback for astropy < 2.0.
        P0 = astropy.constants.atmosphere
    g = astropy.constants.g0
    # Specific gas constant for air.
    R = 287.053 * u.J / (u.kg * u.K)
    # Atmospheric lapse rate.
    L = -6.5e-3 * u.K / u.m
    if airtemp is None:
        # Assume 15C at sea level.
        T0 = (15 * u.deg_C).to(u.K, equivalencies=u.temperature())
    else:
        # Convert airtemp to K.
        T = airtemp.to(u.K, equivalencies=u.temperature())
        # Estimate corresponding sea-level temperature.
        T0 = T - L * elevation
    return P0 * np.power(1 + elevation * L / T0, -g / (L * R))
