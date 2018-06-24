import pytest
import numpy as np

from ..zodiacal import ecl_elong, zodiacal_color_factor


def test_ecl_elong():
    assert ecl_elong(0, 0) == 0
    assert np.array_equal(ecl_elong(np.zeros(3), 0), np.zeros(3))
    assert np.array_equal(ecl_elong(0, np.zeros(2)), np.zeros(2))
    assert np.array_equal(ecl_elong(np.zeros(3), np.zeros((2, 1))), np.zeros((2, 3)))
    assert np.allclose(
        np.round(ecl_elong([10, 20, 30], 40), 3),
        [41.026, 43.958, 48.439])


def test_zodiacal_color_factor():
    assert zodiacal_color_factor(500, 30) == 1
    assert zodiacal_color_factor(500, 90) == 1
    assert np.allclose(
        np.round(zodiacal_color_factor([400, 600, 800], 60), 5),
        [0.89824, 1.05543, 1.14288])
