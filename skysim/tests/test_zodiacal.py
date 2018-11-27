import pytest
import numpy as np

from ..zodiacal import airmass_zodi, ecl_elong, zodiacal_color_factor, \
    get_zodiacal_flux500, zodiacal_scattering, get_zodiacal


def test_airmass_zodi():
    assert np.allclose(airmass_zodi(0), 1)
    assert np.allclose(airmass_zodi(np.zeros(5)), np.ones(5))
    assert np.allclose(
        np.round(airmass_zodi(np.arange(15, 45, 5)), 5),
        [1.03528, 1.06418, 1.10338, 1.15470, 1.22077, 1.30540])


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


def test_get_zodiacal_flux500():
    assert get_zodiacal_flux500(15, 15) == 1860
    assert np.allclose(
        get_zodiacal_flux500([15, 20], [15, 20]), [1860, 910])
    # Values from the table.
    assert np.allclose(
        get_zodiacal_flux500([[15, 15], [20, 20]], [[15, 20], [15, 20]]),
        [[1860,  1110], [1410, 910]])
    # Values requiring interpolation.
    assert np.allclose(
        get_zodiacal_flux500([[16, 16], [19, 19]], [[16, 19], [16, 19]]),
        [[1630, 1210], [1390, 1060]])


def test_zodiacal_scattering():
    assert zodiacal_scattering(10 ** 0) == (-2.692, -2.598)
    assert np.allclose(
        zodiacal_scattering([10 ** 2, 10 ** 3]),
        (np.array([ 0.122,  0.866]), np.array([ 0.02 ,  0.702])))


def test_get_zodiacal():
    assert np.allclose(get_zodiacal(500, 20, 20, 10), 813.49718453)