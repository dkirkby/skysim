"""Compute optical airglow emission and transmission.

Refer to Section 4 of Nolll 2012 for details.
"""
import numpy as np


def airmass_ag(z):
    """Calculate the effective airmass of airglow emission from ~90km.

    Use equation (23) of Noll 2012.

    Parameters
    ----------
    z : float or array
        Zenith angle(s) in degrees.

    Returns
    -------
    float or array
        Airmass(es) corresponding to each input zenith angle.
    """
    z = np.asarray(z)
    return (1 - 0.972 * np.sin(np.deg2rad(z)) ** 2) ** -0.5


def airglow_scattering(z):
    """Calculate net Rayleigh and Mie scattering of airglow emission.

    Use equations (24) and (25) of Noll 2012 to approximate the
    net effect of scattering as an optical depth multiplier.
    Negative values are possible and indicate that scattering
    of indirect airglow into the line of sight exceeds scattering
    of direct airglow out of the line of sight.
    
    Parameters
    ----------
    z : float or array
        Zenith angle(s) in degrees.

    Returns
    -------
    tuple
        Tuple (fR, fM) giving the net optical depth multipliers for
        Rayleigh and Mie scattering, respectively.  The components
        fR, fM will be floats or arrays matching the input z shape.
    """
    x = np.log10(airmass_ag(z))
    # Calculate the Rayleigh scattering fraction.
    fR = 1.669 * x - 0.146
    fM = 1.732 * x - 0.318
    return fR, fM
