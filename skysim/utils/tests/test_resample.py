import pytest
import numpy as np

from ..resample import centers_to_edges, resample_binned, resample_density


def test_centers_to_edges():
    edges = centers_to_edges([1., 2., 3.])
    assert np.allclose(edges, [0.5, 1.5, 2.5, 3.5])


def test_resample_binned():
    assert np.allclose(resample_binned(
        np.arange(4), np.arange(4), np.ones(3)), 1)
    assert np.allclose(resample_binned(
        np.arange(3.5, step=0.5), np.arange(4), np.ones(3)), 0.5)
    assert np.allclose(resample_binned(
        [0., 1., 2.], [0.5, 1.5], [1.]), [0.5, 0.5])
    with pytest.raises(ValueError):
        resample_binned([0., 1., 2.], [0.5, 1.5], [1.], zero_pad=False)


def test_resample_density():
    assert np.allclose(resample_density(
        np.arange(4), np.arange(4), np.ones(4)), 1)
    assert np.allclose(resample_density(
        np.arange(1, 3), np.arange(4), np.ones(4)), 1)
    assert np.allclose(resample_density(
        [0., 1., 2.], [0.5, 1.5], [1., 1.]), [0.5, 1., 0.5])
    with pytest.raises(ValueError):
        resample_density([0., 1., 2.], [0.5, 1.5], [1., 1.], zero_pad=False)


def test_resample_multidim():
    assert np.allclose(
        resample_binned([0, 1, 3], [-1, 1, 3], [[1, 1], [4, 4]], axis=0),
        [[0.5, 0.5], [4., 4.]])
    assert np.allclose(
        resample_binned([0, 1, 3], [-1, 1, 3], [[1, 1], [4, 4]], axis=1),
        [[0.5, 1.], [2., 4.]])
    assert np.allclose(
        resample_binned([0, 1, 3], [-1, 1, 3], [[1, 4], [1, 4]], axis=1),
        [[0.5, 4.], [0.5, 4.]])
    assert np.allclose(
        resample_density([0, 1, 3], [-2, 2], [[0, 1], [0, 1]], axis=0),
        [[0., 1.], [0., 1.], [0., 1.]])
    assert np.allclose(
        resample_density([0, 1, 3], [-2, 2], [[0, 1], [0, 1]], axis=1),
        [[0.5, 1., 1.], [0.5, 1., 1.]])
