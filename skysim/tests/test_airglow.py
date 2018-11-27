import pytest
import numpy as np

from ..airglow import airmass_ag, airglow_scattering, get_airglow


def test_airmass_ag():
    assert airmass_ag(0) == 1
    assert np.array_equal(airmass_ag(np.zeros(5)), np.ones(5))
    assert np.allclose(
        np.round(airmass_ag(np.arange(15, 45, 5)), 5),
        [1.03424, 1.06221, 1.10003, 1.14935, 1.21248, 1.29273])


def test_airglow_scattering():
    assert np.allclose(airglow_scattering(0), [-0.146, -0.318])
    assert np.allclose(
        np.round(airglow_scattering([10, 20, 30]), 3),
        [[-0.135, -0.102, -0.045], [-0.307, -0.273, -0.213]])


def test_get_airglow():
    lam = np.linspace(300, 1100, 5)
    #assert np.allclose(
    #    np.round(get_airglow(lam, 0), 1),
    #    0.)