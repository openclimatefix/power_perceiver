import datetime
from copy import deepcopy

import pytest

from power_perceiver.load_raw.data_sources.raw_gsp_data_source import RawGSPDataSource
from power_perceiver.load_raw.data_sources.raw_pv_data_source import RawPVDataSource
from power_perceiver.load_raw.data_sources.raw_satellite_data_source import RawSatelliteDataSource
from power_perceiver.load_raw.raw_dataset import RawDataset

SAT_HEIGHT_IN_PIXELS = 128
SAT_WIDTH_IN_PIXELS = 256
SAT_N_EXPECTED_TIMESTEPS = 37  # 12 steps of history + 1 for t0 + 24 of forecast

# TODO: Use public data :)
PV_METADATA_FILENAME = "~/data/PV/Passiv/ocf_formatted/v0/system_metadata_OCF_ONLY.csv"
PV_POWER_FILENAME = "~/data/PV/Passiv/ocf_formatted/v0/passiv.netcdf"
N_PV_SYSTEMS_PER_EXAMPLE = 8


def _get_sat_data_source(
    zarr_path=(
        "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/"
        "eumetsat_seviri_hrv_uk.zarr"
    ),
    height_in_pixels=SAT_HEIGHT_IN_PIXELS,
    width_in_pixels=SAT_WIDTH_IN_PIXELS,
    history_duration=datetime.timedelta(hours=1),
    forecast_duration=datetime.timedelta(hours=2),
    start_date="2020-01-01",
    end_date="2020-12-31 12:59",
) -> RawSatelliteDataSource:
    return RawSatelliteDataSource(
        zarr_path=zarr_path,
        roi_height_pixels=height_in_pixels,
        roi_width_pixels=width_in_pixels,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        start_date=start_date,
        end_date=end_date,
    )


@pytest.fixture(scope="session")
def sat_data_source() -> RawSatelliteDataSource:
    return _get_sat_data_source()


# `scope="session"` caches and re-uses the opened `sat_data_opened` so
# we only have to open the Zarr file once!
# But, beware, if any test modifies the return value from `sat_data_opened`
# then that modification will be visible to subsequent tests!
@pytest.fixture(scope="session")
def sat_data_opened() -> RawSatelliteDataSource:
    # Don't use `sat_data_source` fixture, because then the code below would run `per_worker_init`
    # on the "unopened" `sat_data_source` fixture, and so tests which expect an "unopened"
    # RawSatelliteDataSource object would actually get an opened object!
    sat_data_source = _get_sat_data_source()
    sat_data_source.per_worker_init(worker_id=1)
    return sat_data_source


@pytest.fixture(scope="session")
def sat_data_loaded() -> RawSatelliteDataSource:
    # Don't use `sat_data_source` fixture, because that will run `per_worker_init`
    # on the "unopened" `sat_data_source` fixture!
    sat_data_source = _get_sat_data_source()
    sat_data_source.per_worker_init(worker_id=1)

    periods = sat_data_source.get_contiguous_t0_time_periods()
    periods = periods.iloc[:3]
    sat_data_source.load_subset_into_ram(periods)

    return sat_data_source


@pytest.fixture(scope="session")
def pv_data_source() -> RawPVDataSource:
    pv = RawPVDataSource(
        pv_power_filename=PV_POWER_FILENAME,
        pv_metadata_filename=PV_METADATA_FILENAME,
        start_date="2020-01-01",
        end_date="2020-01-03",
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=2),
        roi_height_meters=64_000,
        roi_width_meters=64_000,
        n_pv_systems_per_example=N_PV_SYSTEMS_PER_EXAMPLE,
    )
    pv.per_worker_init(worker_id=0)
    return pv


@pytest.fixture(scope="session")
def gsp_data_source() -> RawGSPDataSource:  # noqa: D103
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


@pytest.fixture(scope="session")
def raw_dataset_with_sat_only(sat_data_source: RawSatelliteDataSource) -> RawDataset:
    return RawDataset(
        data_source_combos=dict(sat_only=(sat_data_source,)),
        ds_combo_for_subsetting="sat_only",
        min_duration_to_load_per_epoch=datetime.timedelta(hours=20),
        n_examples_per_epoch=16,
    )


@pytest.fixture(scope="session")
def raw_dataset_with_sat_only_and_gsp_pv_sat(
    sat_data_source: RawSatelliteDataSource,
    pv_data_source: RawPVDataSource,
    gsp_data_source: RawGSPDataSource,
) -> RawDataset:
    return RawDataset(
        data_source_combos=dict(
            sat_only=(sat_data_source,),
            gsp_pv_sat=(gsp_data_source, pv_data_source, deepcopy(sat_data_source)),
        ),
        ds_combo_for_subsetting="sat_only",
        min_duration_to_load_per_epoch=datetime.timedelta(hours=20),
        n_examples_per_epoch=16,
    )
