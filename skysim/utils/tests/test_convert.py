import numpy as np

import astropy.units as u

from ..convert import radiance_to_sb


def test_radiance_to_sb():
    conv = radiance_to_sb(500, u.Unit('1e-8 W / (m2 sr um)'))
    assert np.allclose(conv, 1690.273517)
    conv = radiance_to_sb(np.full(20, 500), u.Unit('1e-8 W / (m2 sr um)'))
    assert conv.shape == (20,)
    assert np.allclose(conv, 1690.273517)
