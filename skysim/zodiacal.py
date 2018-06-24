"""Compute optical zodiacal scattering and extinction.

See Section 3.3 of Noll 2012 for details.
"""
import numpy as np


def ecl_elong(ecl_lon, ecl_lat):
    """Calculate the elongation for a specified ecliptic longitude and latitude.

    Use equation (11) of Leinert 1998:

    .. math::

        \cos\epsilon = \cos(\lambda - \lambda_\odot) \cos\beta

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