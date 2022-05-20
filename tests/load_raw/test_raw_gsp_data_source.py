import datetime

import numpy as np
import pandas as pd
import pytest

from power_perceiver.consts import Location
from power_perceiver.load_raw.data_sources.raw_gsp_data_source import RawGSPDataSource


@pytest.fixture(scope="module")
def gsp_data_source() -> RawGSPDataSource:
    gsp = RawGSPDataSource(
        gsp_pv_power_zarr_path="~/data/PV/GSP/v3/pv_gsp.zarr",
        gsp_id_to_region_id_filename="~/data/PV/GSP/eso_metadata.csv",
        sheffield_solar_region_path="~/data/PV/GSP/gsp_shape",
        start_date="2020-01-01",
        end_date="2020-06-01",
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=2),
    )
    gsp.per_worker_init(worker_id=0)
    return gsp


def test_init(gsp_data_source: RawGSPDataSource):
    assert np.isfinite(gsp_data_source._data_in_ram.data).all()


def test_get_osgb_location_for_example(gsp_data_source: RawGSPDataSource):
    location = gsp_data_source.get_osgb_location_for_example()
    assert isinstance(location, Location)
    assert isinstance(location.x, float)
    assert isinstance(location.y, float)


def test_get_spatial_slice(gsp_data_source: RawGSPDataSource):
    for gsp_id in gsp_data_source.data_in_ram.gsp_id:
        gsp = gsp_data_source.data_in_ram.sel(gsp_id=gsp_id)
        location = Location(x=gsp.x_osgb.item(), y=gsp.y_osgb.item())
        spatial_slice = gsp_data_source._get_spatial_slice(gsp_data_source.data_in_ram, location)
        assert spatial_slice.gsp_id.item() == gsp.gsp_id.item()
        assert len(spatial_slice.gsp_id) == 1


def test_get_example_and_empty_example(gsp_data_source: RawGSPDataSource):
    periods = gsp_data_source.get_contiguous_t0_time_periods()
    period = periods.iloc[0]
    date_range = pd.date_range(period.start_dt, period.end_dt, freq="5T")
    rng = np.random.default_rng(seed=42)
    for t0 in rng.choice(date_range, size=288):
        location = gsp_data_source.get_osgb_location_for_example()
        xr_example = gsp_data_source.get_example(t0_datetime_utc=t0, center_osgb=location)
        gsp_data_source.check_xarray_data(xr_example)
        np.testing.assert_array_equal(xr_example.shape, gsp_data_source.empty_example.shape)
        np_example = RawGSPDataSource.to_numpy(xr_example)
        RawGSPDataSource.check_numpy_data(np_example)
