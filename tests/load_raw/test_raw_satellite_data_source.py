import datetime

import pandas as pd
import pytest

from power_perceiver.consts import Location
from power_perceiver.load_raw.data_sources.raw_satellite_data_source import RawSatelliteDataSource

# for the moment
SAT_HEIGHT_IN_PIXELS = 128
SAT_WIDTH_IN_PIXELS = 256


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_init(sat_data_source):
    assert sat_data_source._data_in_ram is None
    assert sat_data_source._data_on_disk is None


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_init_start_and_end_dates_swapped(get_sat_data_source):
    with pytest.raises(AssertionError):
        get_sat_data_source(
            end_date=datetime.datetime(year=2020, month=1, day=1),
            start_date=datetime.datetime(year=2020, month=12, day=31, hour=23, minute=59),
        )


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_per_worker_init(sat_data_opened):
    assert sat_data_opened._data_on_disk is not None


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_datetimes(sat_data_opened):
    dt_index = sat_data_opened.datetime_index
    # Test that we don't have any datetimes at midnight (00:00 to 00:59):
    assert not (dt_index.hour == 0).any()


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_get_spatial_slice(sat_data_opened: RawSatelliteDataSource):
    # Select a location roughly in the middle of the Satellite imagery:
    location_center_osgb = Location(x=66400, y=357563)
    selection = sat_data_opened._get_spatial_slice(
        sat_data_opened._data_on_disk,
        location_center_osgb,
    )
    assert len(selection.x_geostationary) == SAT_WIDTH_IN_PIXELS
    assert len(selection.y_geostationary) == SAT_HEIGHT_IN_PIXELS

    with pytest.raises(AssertionError):
        selection = sat_data_opened._get_spatial_slice(
            sat_data_opened._data_on_disk,
            # Try getting an image centered on the left edge:
            Location(x=-1371057, y=357563),
        )


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_load_subset_into_ram(sat_data_loaded: RawSatelliteDataSource):
    assert sat_data_loaded._data_in_ram is not None


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_get_example(sat_data_loaded: RawSatelliteDataSource):
    xr_example = sat_data_loaded.get_example(
        t0_datetime_utc=pd.Timestamp("2020-01-01 12:00"),
        center_osgb=Location(x=66400, y=357563),
    )
    del xr_example  # TODO: Do something with this!


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_get_osgb_location_for_example(sat_data_loaded: RawSatelliteDataSource):
    location = sat_data_loaded.get_osgb_location_for_example()
    print("LOCATION!")
    print(location)
    for coord in location:
        assert isinstance(coord, float), f"{type(coord)=}"
    # TODO: Check the OSGB coords are sane!
