"""Compute optical zodiacal scattering and extinction.

See Section 3.3 of Noll 2012 for details.
"""
import numpy as np

import scipy.interpolate

import astropy.constants
import astropy.units as u

import skysim.utils.resample
import skysim.utils.data
import skysim.utils.convert
import skysim.transmission


def airmass_zodi(zenith_angle):
    """Calculate the airmass used for zodiacal light calculations.

    Use equation (2) of Noll 2012, which is originally from
    Rozenberg 1966.

    Parameters
    ----------
    z : float or array
        Zenith angle(s) in degrees.

    Returns
    -------
    float or array
        Airmass(es) corresponding to each input zenith angle.
    """
    cosz = np.cos(np.radians(zenith_angle))
    return (cosz + 0.025 * np.exp(-11 * cosz)) ** -1


def ecl_elong(ecl_lon, ecl_lat):
    """Calculate the elongation for a specified ecliptic longitude and latitude.

    Use equation (11) of Leinert 1998:

    .. math::

        \\cos\\epsilon = \\cos(\\lambda - \\lambda_\\odot) \\cos\\beta

    Parameters
    ----------
    ecl_lon : float or array
        Heliocentric ecliptic longitude in degrees.
    ecl_lat : float or array
        Ecliptic latitude in degrees.

    Returns
    -------
    float or array
        Ecliptic elongation in degrees.
    """
    cos_elong = np.cos(np.deg2rad(ecl_lon)) * np.cos(np.deg2rad(ecl_lat))
    return np.rad2deg(np.arccos(cos_elong))


def zodiacal_color_factor(lam, elong):
    """Calculate the redenning of the solar spectrum.

    Interpolate in ecliptic longitude between the color factors calculated
    with equation (22) of Leinert 1998, as described in Section 8.4.2.

    Parameters
    ----------
    lam : float or array
        Observed wavelength in nm.
    elong : float or array
        Ecliptic elongation angle in degrees, which can be obtained using
        :func:`ecl_elong`.

    Returns
    -------
    float or array
        The color correction factor(s) to apply to the solar spectrum.
    """
    lam = np.atleast_1d(lam)
    lo = lam <= 500
    hi = lam > 500
    # Note that "log" in Leinert means log10.
    logr = np.log10(lam / 500.)
    # Calculate the factors at eps=30, 90 using eqn (22).
    f30 = np.empty(lam.shape)
    f30[lo] = 1.0 + 1.2 * logr[lo]
    f30[hi] = 1.0 + 0.8 * logr[hi]
    f90 = np.empty(lam.shape)
    f90[lo] = 1.0 + 0.9 * logr[lo]
    f90[hi] = 1.0 + 0.6 * logr[hi]
    # Interpolate linearly in elongation angles between 30-90 deg.
    r = np.clip((elong - 30.) / 60., 0., 1.)
    return (1 - r) * f30 + r * f90


# Data from Table 17 of Leinert 1998.
_zodi_beta = np.array([0, 5, 10, 15, 20, 25, 30, 45, 60, 75])
_zodi_lon = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 60,
                      75, 90, 105, 120, 135, 150, 165, 180])
_zodi_flux = np.array(
    [
        [0,        0,    0, 3140, 1610, 985, 640, 275, 150, 100],
        [0,        0,    0, 2940, 1540, 945, 625, 271, 150, 100],
        [0,        0, 4740, 2470, 1370, 865, 590, 264, 148, 100],
        [11500, 6780, 3440, 1860, 1110, 755, 525, 251, 146, 100],
        [6400,  4480, 2410, 1410,  910, 635, 454, 237, 141,  99],
        [3840,  2830, 1730, 1100,  749, 545, 410, 223, 136,  97],
        [2480,  1870, 1220,  845,  615, 467, 365, 207, 131,  95],
        [1650,  1270,  910,  680,  510, 397, 320, 193, 125,  93],
        [1180,   940,  700,  530,  416, 338, 282, 179, 120,  92],
        [910,    730,  555,  442,  356, 292, 250, 166, 116,  90],
        [505,    442,  352,  292,  243, 209, 183, 134, 104,  86],
        [338,    317,  269,  227,  196, 172, 151, 116,  93,  82],
        [259,    251,  225,  193,  166, 147, 132, 104,  86,  79],
        [212,    210,  197,  170,  150, 133, 119,  96,  82,  77],
        [188,    186,  177,  154,  138, 125, 113,  90,  77,  74],
        [179,    178,  166,  147,  134, 122, 110,  90,  77,  73],
        [179,    178,  165,  148,  137, 127, 116,  96,  79,  72],
        [196,    192,  179,  165,  151, 141, 131, 104,  82,  72],
        [230,    212,  195,  178,  163, 148, 134, 105,  83,  72]
    ])
_zodi_interpolator = None


def get_zodiacal_flux500(ecl_lon, ecl_lat):
    """Return zodical flux at 500nm incident above the atmosphere.

    Automatically broadcasts over ecl_lon and ecl_lat. Return
    value is scalar when both inputs are scalar.

    Parameters
    ----------
    ecl_lon : float or array
        Heliocentric ecliptic longitude in degrees.
    ecl_lat : float or array
        Ecliptic latitude in degrees.

    Returns
    -------
    float or array
        Zodiacal flux at 500nm incident above the atmosphere, in units of
        1e-8 W / (m2 sr um).
    """
    ecl_lon = np.asarray(ecl_lon)
    ecl_lat = np.asarray(ecl_lat)
    scalar_input = np.isscalar(ecl_lon + ecl_lat)
    # Wrap lon to distance from lon=0 in the range [0, 180].
    ecl_lon = np.fmod(np.abs(ecl_lon), 360)
    wrap = np.floor(ecl_lon / 180)
    ecl_lon = wrap * (360 - ecl_lon) + (1 - wrap) * ecl_lon
    assert np.all((ecl_lon >= 0) & (ecl_lon <= 180))
    ecl_lat = np.abs(ecl_lat)
    if not np.all(ecl_lat < 90):
        raise ValueError('Expected ecl_lat in (-90, 90).')
    global _zodi_interpolator
    if _zodi_interpolator is None:
        _zodi_interpolator = scipy.interpolate.RectBivariateSpline(
            _zodi_lon, _zodi_beta, _zodi_flux, kx=1, ky=1)
    flux500 = _zodi_interpolator(ecl_lon, ecl_lat, grid=False)
    return np.float(flux500) if scalar_input else flux500


def zodiacal_scattering(I0):
    """Calculate net Rayleigh and Mie scattering of zodiacal light.

    Use equations (19) and (20) of Noll 2012 to approximate the
    net effect of scattering as an optical depth multiplier.
    Negative values are possible and indicate that scattering
    of indirect zodiacal light into the line of sight exceeds scattering
    of direct zodiacal light out of the line of sight.

    Parameters
    ----------
    I0 : float or array
        Zodiacal flux in 1e-8 W / (m2 sr um).

    Returns
    -------
    tuple
        Tuple (fR, fM) giving the net optical depth multipliers for
        Rayleigh and Mie scattering, respectively.  The components
        fR, fM will be floats or arrays matching the input z shape.
    """
    x = np.log10(np.atleast_1d(I0))
    fR = np.empty_like(x)
    lo = x <= 2.244
    hi = ~lo
    fR[lo] = 1.407 * x[lo] - 2.692
    fR[hi] = 0.527 * x[hi] - 0.715
    fM = np.empty_like(x)
    lo = x <= 2.255
    hi = ~lo
    fM[lo] = 1.309 * x[lo] - 2.598
    fM[hi] = 0.468 * x[hi] - 0.702

    return fR, fM


def get_zodiacal(lam, ecl_lon, ecl_lat, z, p=744., H=2.64,
                 redden=True, Rayleigh=True, Mie=True, absorption=True):
    """Calculate zodiacal flux.

    Parameters
    ----------
    lam : float or array
        Wavelength in nanometers.
    ecl_lon : float or array
        Heliocentric ecliptic longitude in degrees.
    ecl_lat : float or array
        Ecliptic latitude in degrees.
    z : float or array
        Zenith angle in degrees, used to calculate airmass.
    p : float or array
        Pressure at the observation elevation in hPa, used for Rayleigh
        scattering.
    H : float or array
        Elevation of the observation in km, used for Rayleigh scattering.
    redden : bool
        Apply redenning of solar spectrum.
    Rayleigh : bool
        Apply Rayleigh scattering effects.
    Mie : bool
        Apply aerosol Mie scattering effects.
    absorption : bool
        Apply molecular (but not ozone) absorption effects.

    Returns
    -------
    float or array
        Array of radiances corresponding to each input wavelength,
        in units of ph / (arcsec2 m2 s nm).
    """
    lam = np.atleast_1d(lam)
    # Get the solar flux in 1e-8 W / (m2 sr um).
    sol = skysim.utils.data.get('solarspec')
    sol_lam = sol['wavelength'].data
    incident_flux = sol['flux'].data * skysim.utils.convert.radiance_to_flux(
        sol_lam, u.Unit('1e-8 W / (m2 sr um)'))
    if redden:
        # Apply the redenning of the solar spectrum at this (ecl_lon, ecl_lat).
        incident_flux *= zodiacal_color_factor(
            sol_lam, ecl_elong(ecl_lon, ecl_lat))
    # Calculate incident surface brightness at 500nm in 1e-8 W / (m2 sr um).
    flux500 = get_zodiacal_flux500(ecl_lon, ecl_lat)
    # Normalize the redenned solar spectrum.
    incident_flux *= flux500 / np.interp(500., sol_lam, incident_flux)
    # Resample to the output wavelength grid.
    incident_flux = skysim.utils.resample.resample_density(
        lam, sol_lam, incident_flux)
    # Calculate scattering and absorption optical depths.
    tau0 = np.zeros_like(incident_flux)
    if Rayleigh or Mie:
        # Calculate the scattering fractions.
        fR, fM = zodiacal_scattering(incident_flux)
    if Rayleigh:
        tau0 += fR * skysim.transmission.tau0R(lam, p, H)
    if Mie:
        tau0 += fM * skysim.transmission.tau0M(lam)
    if absorption:
        tau0 += skysim.transmission.tau0ma(lam)
    X = airmass_zodi(z)
    # Calulate the flux reaching the surface.
    surface_flux = incident_flux * np.exp(-tau0 * X)
    # Convert to radiance at the surface in ph / (arcsec2 m2 s nm).
    return surface_flux / skysim.utils.convert.radiance_to_flux(
        lam, u.Unit('1e-8 W / (m2 sr um)'))
