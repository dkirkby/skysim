import pytest
import numpy as np

import astropy.coordinates
import astropy.time
import astropy.utils
import astropy.units as u

from ..meta import get_sky_metadata


def test_meta():
    # Use bundled IERS-B instead of downloading IERS-A.
    from astropy.utils import iers
    iers.conf.auto_download = False
    where = astropy.coordinates.EarthLocation.from_geocentric(
        x=-1463969.30185172, y=-5166673.34223433, z=3434985.71204565, unit='m')
    when = astropy.time.Time('2017-01-01T00:00:00', format='isot')
    what = astropy.coordinates.SkyCoord(
        ra=343 * u.deg, dec=15 * u.deg, frame='icrs')
    M = get_sky_metadata(
        location=where, obstime=when, pointing=what, pressure=0)
    assert np.allclose(M[0]['solar_flux'], 77.37)
