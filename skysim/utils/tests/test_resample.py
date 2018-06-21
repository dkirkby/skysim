import pytest
import numpy as np

from ..resample import *


def test_blah():
    edges = centers_to_edges([1., 2., 3.])
    assert np.allclose(edges, [0.5, 1.5, 2.5, 3.5])