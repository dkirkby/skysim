import pytest
import numpy as np

from ..transmission import tau0R


def test_tau0R():
    assert np.allclose(tau0R(0.5, 1000., 1.0), 0.14164404017195453)
    lam = np.linspace(0.3, 1.1, 5)
    assert np.allclose(
        np.round(tau0R(lam, 1000., 1.0), 5),
        [1.19528, 0.14164, 0.03604, 0.01306, 0.0058])
    assert np.array_equal(
        tau0R(lam, 1000., 1.0), tau0R(lam, 1000. * np.ones(5), 1.0))
    with pytest.raises(ValueError):
        tau0R(lam, 1000. * np.ones(2), 1.0)
    assert tau0R(lam, 1000. * np.ones((2, 1)), 1.0).shape == (2, 5)
