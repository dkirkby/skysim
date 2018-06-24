import pytest
import numpy as np

from ..data import get


def test_get_solarspec():
    t1 = get('solarspec')
    assert t1.colnames == ['wavelength', 'flux']
    assert t1['wavelength'].unit == 'nm'
    assert t1['flux'].unit == 'ph / (s cm2 nm)'
    t2 = get('solarspec')
    assert t2 is t1
    t3 = get('solarspec', force_load=True)
    assert not (t3 is t1)


def test_get_atmosphere():
    t1 = get('atmosphere')
    assert t1.colnames == [
        'wavelength', 'trans_ma', 'trans_o3', 'airglow_cont', 'airglow_line']
    assert t1['wavelength'].unit == 'nm'


def test_get_invalid():
    with pytest.raises(ValueError):
        get('invalid')