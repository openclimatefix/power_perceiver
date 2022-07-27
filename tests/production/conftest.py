import numpy as np
import pytest

from power_perceiver.production.batch import GSP, NWP, PV, HRVSatellite


@pytest.fixture()
def pv():

    batch_size = 4
    timesteps = 3
    pv_systems = 2

    pv = np.random.random([batch_size, timesteps, pv_systems])
    row_number = np.random.random([batch_size, pv_systems])
    pv_y_osgb = np.random.random([batch_size, pv_systems])
    pv_x_osgb = np.random.random([batch_size, pv_systems])
    pv_time_utc = np.random.random([batch_size, timesteps])

    return PV(
        pv=pv,
        pv_t0_idx=10,
        pv_system_row_number=row_number,
        pv_y_osgb=pv_y_osgb,
        pv_x_osgb=pv_x_osgb,
        pv_time_utc=pv_time_utc,
    )


@pytest.fixture()
def gsp():

    batch_size = 4
    timesteps = 3

    gsp = np.random.random([batch_size, timesteps, 1])
    gsp_id = np.random.random([batch_size])
    gsp_y_osgb = np.random.random([batch_size])
    gsp_x_osgb = np.random.random([batch_size])
    gsp_time_utc = np.random.random([batch_size, timesteps])

    return GSP(
        gsp=gsp,
        gsp_t0_idx=10,
        gsp_id=gsp_id,
        gsp_y_osgb=gsp_y_osgb,
        gsp_x_osgb=gsp_x_osgb,
        gsp_time_utc=gsp_time_utc,
    )


@pytest.fixture()
def nwp():

    batch_size = 4
    timesteps = 3
    channels = 2
    x_steps = 5
    y_steps = 6

    nwp = np.random.random([batch_size, timesteps, channels, y_steps, x_steps])
    nwp_target_time_utc = np.random.random([batch_size, timesteps])
    nwp_y_osgb = np.random.random([batch_size, y_steps])
    nwp_x_osgb = np.random.random([batch_size, x_steps])
    nwp_init_time_utc = np.random.random([batch_size, timesteps])

    return NWP(
        nwp=nwp,
        nwp_t0_idx=10,
        nwp_target_time_utc=nwp_target_time_utc,
        nwp_y_osgb=nwp_y_osgb,
        nwp_x_osgb=nwp_x_osgb,
        nwp_init_time_utc=nwp_init_time_utc,
    )


@pytest.fixture()
def hrv_satellite():

    batch_size = 4
    timesteps = 3
    channels = 2
    x_steps = 5
    y_steps = 6

    hrvsatellite_actual = np.random.random([batch_size, timesteps, channels, y_steps, x_steps])
    hrvsatellite_x_geostationary = np.random.random([batch_size, y_steps, x_steps])
    hrvsatellite_y_geostationary = np.random.random([batch_size, y_steps, x_steps])
    hrvsatellite_y_osgb = np.random.random([batch_size, y_steps, x_steps])
    hrvsatellite_x_osgb = np.random.random([batch_size, y_steps, x_steps])
    hrvsatellite_time_utc = np.random.random([batch_size, timesteps])

    return HRVSatellite(
        hrvsatellite_actual=hrvsatellite_actual,
        hrvsatellite_x_geostationary=hrvsatellite_y_geostationary,
        hrvsatellite_y_geostationary=hrvsatellite_x_geostationary,
        hrvsatellite_y_osgb=hrvsatellite_y_osgb,
        hrvsatellite_x_osgb=hrvsatellite_x_osgb,
        hrvsatellite_time_utc=hrvsatellite_time_utc,
        hrvsatellite_t0_idx=10,
    )
