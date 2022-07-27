import numpy as np
import pytest

from power_perceiver.production.batch import PV


def test_pv(pv):
    pass


def test_pv_check_negative(pv):
    with pytest.raises(Exception):
        pv_x_osgb = pv.pv_x_osgb
        pv_x_osgb[0, 0] = -1
        pv.pv_x_osgb = pv_x_osgb
