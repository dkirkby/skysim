"""Compute scattering and absorption effects on transmission.

Refer to Section 2 of Noll 2012.
"""
import numpy as np


def tau0R(lam, p=744., H=2.64):
    """Calculate zenith optical depth due to Rayleigh scattering.

    Use equation (3) of Noll 2012:

    .. math::

        \tau_0^R(\lambda) = \frac{p}{1013.25\,\text{hPa}} \left(
            0.00864 + 6.5\times 10^{-6} \frac{H}{1\,\text{km}}\right)
        \lambda^{-(0.3916 + 0.074 \lambda + 0.050 / \lambda)} \; .

    Automatically broadcasts over any input arrays.

    Parameters
    ----------
    lam : float or array
        Wavelength in microns.
    p : float or array
        Pressure at the observation elevation in hPa.
    H : float or array
        Elevation of the observation in km.

    Returns
    -------
    float or array
        Optical depth(s) calculated for the input parameters.
    """
    lam = np.asarray(lam)
    p = np.asarray(p)
    H = np.asarray(H)
    return (p / 1013.25) * (0.00864 + 6.5e-6 * H) * np.power(
        lam, -(3.916 + 0.074 * lam + 0.050 / lam))


def tau0M(lam, lam0=0.4, k0=0.013, alpha=-1.38):
    """Calculate zenith optical depth due to Mie scattering off aerosols.

    Use equation (4) of Noll 2012:

    .. math::

        k^M(\lambda) = k_0 \lambda^\alpha

    Parameters
    ----------
    lam : float or array
        Wavelength in microns.
    lam0 : float
        Optical depth is constant below this wavelength in microns.
    k0 : float
        Extinction at 1 micron in mag / airmass.
    alpha : float
        Extinction wavelength power.
    """
    lam = np.asarray(lam)
    return 0.4 * np.log(10) * k0 * np.power(np.maximum(lam, lam0), alpha)
