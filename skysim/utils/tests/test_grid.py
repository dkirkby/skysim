import numpy as np

import astropy.coordinates
import astropy.time

from ..grid import AltAzGrid


def test_grid():
    where = astropy.coordinates.EarthLocation.from_geocentric(
        x=-1463969.30185172, y=-5166673.34223433, z=3434985.71204565, unit='m')
    when = astropy.time.Time('2018-01-01T00:00:00', format='isot')
    G = AltAzGrid(min_alt=30)
    assert np.all(G.alt >= 25)
    G.location = where
    G.obstime = when
    assert np.all(np.abs(G.dec) <= 90)
    assert np.all(np.abs(G.ecl_lat) <= 90)
