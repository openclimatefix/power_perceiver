import pytest

from power_perceiver.production.batch import get_batch


def test_pv(pv):
    pass


def test_gsp(gsp):
    pass


def test_nwp(nwp):
    pass


def test_hrv_satellite(hrv_satellite):
    pass


def test_pv_check_negative(pv):
    with pytest.raises(Exception):
        pv_x_osgb = pv.pv_x_osgb
        pv_x_osgb[0, 0] = -1
        pv.pv_x_osgb = pv_x_osgb


def test_pv_check_wrong_shape(pv):
    with pytest.raises(Exception):
        pv_x_osgb = pv.pv_x_osgb
        pv_x_osgb = pv_x_osgb[0]
        pv.pv_x_osgb = pv_x_osgb


def test_make_batch(pv, nwp, gsp, hrv_satellite):
    _ = get_batch(pv=pv, nwp=nwp, gsp=gsp, hrvsatelle=hrv_satellite)
