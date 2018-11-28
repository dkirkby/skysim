import numpy as np

import astropy.units as u

from ..convert import radiance_to_flux


def test_radiance_to_flux():
    conv = radiance_to_flux(500, u.Unit('1e-8 W / (m2 sr um)'))
    assert np.allclose(conv, 1690.273517)
    conv = radiance_to_flux(np.full(20, 500), u.Unit('1e-8 W / (m2 sr um)'))
    assert conv.shape == (20,)
    assert np.allclose(conv, 1690.273517)
