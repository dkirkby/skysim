import pytest
import numpy as np

from ..transmission import tau0R, tau0M


def test_tau0R():
    assert np.allclose(tau0R(500, 1000., 1.0), 0.14164404017195453)
    lam = np.linspace(300, 1100, 5)
    assert np.allclose(
        np.round(tau0R(lam, 1000., 1.0), 5),
        [1.19528, 0.14164, 0.03604, 0.01306, 0.0058])
    assert np.array_equal(
        tau0R(lam, 1000., 1.0), tau0R(lam, 1000. * np.ones(5), 1.0))
    with pytest.raises(ValueError):
        tau0R(lam, 1000. * np.ones(2), 1.0)
    assert tau0R(lam, 1000. * np.ones((2, 1)), 1.0).shape == (2, 5)


def test_tau0M():
    assert np.allclose(tau0M(500), 0.031163083715190009)
    lam = np.linspace(300, 1100, 5)
    assert np.allclose(
        np.round(tau0M(lam), 5),
        [0.0424, 0.03116, 0.01959, 0.01385, 0.01050])
