"""Calculate metadata relevant to sky simulations.
"""
import numpy as np

import astropy.coordinates
import astropy.table
import astropy.units as u


def get_sky_metadata(obstime, pointing, location, pressure=None,
                     airtemp=None, ephemeris='builtin'):
    """Calculate sky metadata for specified time and pointing.

    Parameters
    ----------
    obstime : astropy.time.Time
        One or more observing times when metadata should be calculated.
    pointing : astropy.coordinates.SkyCoord
        The telescope pointing at each input time. Must have the same
        shape as ``obstime``.
    location : astropy.coordinates.earth.EarthLocation
        The observatory earth location.
    pressure : astropy.units.Quantity or None
        Atmospheric pressure to use for refraction corrections. When None, use
        :func:`nominal_pressure`. Set to zero for no refraction corrections.
    airtemp : astropy.units.Quantity or None
        Air temperature to use for refraction corrections. Ignored when
        pressure is zero.
    ephemeris : str
        Ephemeris to use for the moon.

    Returns
    -------
    astropy.table.Table
        Table of computed sky metadata for each (time, pointing) observed from
        the specified location.
    """
    if obstime.shape != pointing.shape:
        raise ValueError('Arrays obstime and pointing have different shapes.')
    output = astropy.table.Table()
    # Calculate apparent local sidereal time in degrees.
    output['lst'] = astropy.table.Column(
        np.atleast_1d(obstime.sidereal_time(
            'apparent', longitude=location.lon).to(u.deg).value),
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
    # Estimate 10.7cm solar flux with zero lag.
    # output['solar_flux'] = np.interp(
    #    obstime.mjd, solar['MJD'].data, solar['FLUX'].data)

    return output
