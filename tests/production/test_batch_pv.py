import numpy as np
import pytest

from power_perceiver.production.batch import PV


def test_pv():

    batch_size = 4
    timesteps = 3
    pv_systems = 2

    pv = np.random.random([batch_size, timesteps, pv_systems])
    row_number = np.random.random([batch_size, pv_systems])
    pv_y_osgb = np.random.random([batch_size, pv_systems])
    pv_x_osgb = np.random.random([batch_size, pv_systems])
    pv_time_utc = np.random.random([batch_size, timesteps])

    _ = PV(
        pv=pv,
        pv_t0_idx=10,
        pv_system_row_number=row_number,
        pv_y_osgb=pv_y_osgb,
        pv_x_osgb=pv_x_osgb,
        pv_time_utc=pv_time_utc,
    )


def test_pv_check_negative():

    batch_size = 4
    timesteps = 3
    pv_systems = 2

    pv = np.random.random([batch_size, timesteps, pv_systems])
    row_number = np.random.random([batch_size, pv_systems])
    pv_y_osgb = np.random.random([batch_size, pv_systems])
    pv_x_osgb = np.random.random([batch_size, pv_systems])
    pv_time_utc = np.random.random([batch_size, timesteps])

    pv_x_osgb[0, 0] = -1

    with pytest.raises(Exception):
        _ = PV(
            pv=pv,
            pv_t0_idx=10,
            pv_system_row_number=row_number,
            pv_y_osgb=pv_y_osgb,
            pv_x_osgb=pv_x_osgb,
            pv_time_utc=pv_time_utc,
        )
