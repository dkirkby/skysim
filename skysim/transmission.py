"""Compute scattering and absorption effects on transmission.

Refer to Section 2 of Noll 2012 for details.
"""
import numpy as np

import skysim.utils.resample


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
        Wavelength in nanometers.
    p : float or array
        Pressure at the observation elevation in hPa.
    H : float or array
        Elevation of the observation in km.

    Returns
    -------
    float or array
        Optical depth(s) calculated for the input parameters.
    """
    lam = 1e-3 * np.asarray(lam)  # convert from nm to um
    p = np.asarray(p)
    H = np.asarray(H)
    return (p / 1013.25) * (0.00864 + 6.5e-6 * H) * np.power(
        lam, -(3.916 + 0.074 * lam + 0.050 / lam))


def tau0M(lam, lam0=400., k0=0.013, alpha=-1.38):
    """Calculate zenith optical depth due to Mie scattering off aerosols.

    Use equation (4) of Noll 2012:

    .. math::

        k^M(\lambda) = k_0 \lambda^\alpha

    Parameters
    ----------
    lam : float or array
        Wavelength in nanometers.
    lam0 : float
        Optical depth is constant below this wavelength in nanometers.
    k0 : float
        Extinction at 1000nm in mag / airmass.
    alpha : float
        Extinction wavelength power.
    """
    lam = 1e-3 * np.asarray(lam)  # convert from nm to um
    return 0.4 * np.log(10) * k0 * np.power(np.maximum(lam, 1e-3 * lam0), alpha)


def tau0ma(lam):
    """Calculate zenith optical depth of molecular absorption.
    
    The main absorbers are the molecular oxygen bands (A ~762nm, B ~688nm,
    gamma ~628nm) and water vapor bands (~720nm, 820nm, 940nm).
    See Section 2 and Fig.2 of Noll 2012 for details.
    
    Narrow asorption features are resampled to the requested wavelength
    grid using a flux-conserving algorithm.
    
    Parameters
    ----------
    lam : float or array
        Wavelength in microns.

    Returns
    -------
    float or array
        Zenith optical depth of molecular absorption.
    """
    atm = skysim.utils.data.get('atmosphere')
    transmission = skysim.utils.resample.resample_density(
        lam, atm['wavelength'].data, atm['trans_ma'].data)
    return -np.log(transmission)


def tau0o3(lam):
    """Calculate zenith optical depth of ozone absorption.
    
    The main features are the Huggins band in the near-UV and the
    broad Chappuis bands around 600nm.
    See Section 2 and Fig.2 of Noll 2012 for details.
    
    Narrow asorption features are resampled to the requested wavelength
    grid using a flux-conserving algorithm.
    
    Parameters
    ----------
    lam : float or array
        Wavelength in microns.

    Returns
    -------
    float or array
        Zenith optical depth of molecular absorption.
    """
    atm = skysim.utils.data.get('atmosphere')
    transmission = skysim.utils.resample.resample_density(
        lam, atm['wavelength'].data, atm['trans_o3'].data)
    return -np.log(transmission)
