import datetime

import pytest

from power_perceiver.consts import Location
from power_perceiver.load_raw.data_sources.raw_satellite_data_source import RawSatelliteDataSource


def _get_sat_data_source(
    zarr_path="gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr",
    height_in_pixels=128,
    width_in_pixels=256,
    history_duration=datetime.timedelta(hours=1),
    forecast_duration=datetime.timedelta(hours=2),
    start_date=datetime.datetime(year=2020, month=1, day=1),
    end_date=datetime.datetime(year=2020, month=12, day=31, hour=23, minute=59),
) -> RawSatelliteDataSource:
    return RawSatelliteDataSource(
        zarr_path=zarr_path,
        height_in_pixels=height_in_pixels,
        width_in_pixels=width_in_pixels,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        start_date=start_date,
        end_date=end_date,
    )


@pytest.fixture(scope="module")
def sat_data_source() -> RawSatelliteDataSource:
    return _get_sat_data_source()


# `scope="module"` caches and re-uses the opened `sat_data_opened` so
# we only have to open the Zarr file once!
# But, beware, if any test modifies the return value from `sat_data_opened`
# then that modification will be visible to subsequent tests!
@pytest.fixture(scope="module")
def sat_data_opened() -> RawSatelliteDataSource:
    # Don't use `sat_data_source` fixture, because that will run `per_worker_init`
    # on the "unopened" `sat_data_source` fixture!
    sat_data_source = _get_sat_data_source()
    sat_data_source.per_worker_init(worker_id=1)
    return sat_data_source


def test_init(sat_data_source):
    assert sat_data_source._data_in_ram is None
    assert sat_data_source._data_on_disk is None


def test_init_start_and_end_dates_swapped():
    with pytest.raises(AssertionError):
        _get_sat_data_source(
            end_date=datetime.datetime(year=2020, month=1, day=1),
            start_date=datetime.datetime(year=2020, month=12, day=31, hour=23, minute=59),
        )


def test_per_worker_init(sat_data_opened):
    assert sat_data_opened._data_on_disk is not None


def test_datetimes(sat_data_opened):
    dt_index = sat_data_opened.datetime_index
    # Test that we don't have any datetimes at midnight (00:00 to 00:59):
    assert not (dt_index.hour == 0).any()


def test_get_spatial_slice(sat_data_opened: RawSatelliteDataSource):
    location_center_osgb = Location(x=66400, y=357563)
    selection = sat_data_opened._get_spatial_slice(
        sat_data_opened._data_on_disk,
        location_center_osgb,
    )
