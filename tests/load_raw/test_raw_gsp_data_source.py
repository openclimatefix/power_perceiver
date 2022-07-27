import numpy as np
import pandas as pd
import pytest

from power_perceiver.consts import Location
from power_perceiver.load_raw.data_sources.raw_gsp_data_source import RawGSPDataSource


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_init(gsp_data_source: RawGSPDataSource):  # noqa: D103
    assert np.isfinite(gsp_data_source._data_in_ram.data).all()


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_get_osgb_location_for_example(gsp_data_source: RawGSPDataSource):  # noqa: D103
    location = gsp_data_source.get_osgb_location_for_example()
    assert isinstance(location, Location)
    assert isinstance(location.x, float)
    assert isinstance(location.y, float)


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_get_spatial_slice(gsp_data_source: RawGSPDataSource):  # noqa: D103
    for gsp_id in gsp_data_source.data_in_ram.gsp_id:
        gsp = gsp_data_source.data_in_ram.sel(gsp_id=gsp_id)
        location = Location(x=gsp.x_osgb.item(), y=gsp.y_osgb.item())
        spatial_slice = gsp_data_source._get_spatial_slice(gsp_data_source.data_in_ram, location)
        assert spatial_slice.gsp_id.item() == gsp.gsp_id.item()
        assert len(spatial_slice.gsp_id) == 1


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_get_example_and_empty_example(gsp_data_source: RawGSPDataSource):  # noqa: D103
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
        for batch_key, array in np_example.items():
            assert np.isfinite(array).all(), f"{batch_key=} has non-finite values!"
