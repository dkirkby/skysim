"""Unit conversion utilities.
"""
import numpy as np

import astropy.constants
import astropy.units as u


_cache = {}
_flux = None
_lam = np.arange(300, 11050, 50)

radiance_unit = u.Unit('ph / (arcsec2 m2 s nm)')


def radiance_to_sb(lam, sb_unit):
    """Convert radiance units ph / (arcsec2 m2 s nm) to surface-brightness units.
    """
    global _lam, _flux, _cache

    if _flux is None:
        hc = astropy.constants.h * astropy.constants.c
        _flux = hc / (_lam * u.nm) * radiance_unit / u.ph

    if sb_unit not in _cache:
        _cache[sb_unit] = _flux.to(sb_unit).value

    return np.interp(lam, _lam, _cache[sb_unit])
