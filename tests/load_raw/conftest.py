import datetime
from copy import deepcopy
from datetime import timedelta

import pandas as pd
import pytest

from power_perceiver.load_raw.data_sources.raw_gsp_data_source import RawGSPDataSource
from power_perceiver.load_raw.data_sources.raw_nwp_data_source import RawNWPDataSource
from power_perceiver.load_raw.data_sources.raw_pv_data_source import RawPVDataSource
from power_perceiver.load_raw.data_sources.raw_satellite_data_source import RawSatelliteDataSource
from power_perceiver.load_raw.raw_dataset import RawDataset

SAT_HEIGHT_IN_PIXELS = 128
SAT_WIDTH_IN_PIXELS = 256
SAT_N_EXPECTED_TIMESTEPS = 37  # 12 steps of history + 1 for t0 + 24 of forecast
N_EXAMPLES_PER_BATCH = 16

USE_LOCAL_FILES = True

# TODO: Use public data :)
PV_METADATA_FILENAME = "~/data/PV/Passiv/ocf_formatted/v0/system_metadata_OCF_ONLY.csv"
PV_POWER_FILENAME = "~/data/PV/Passiv/ocf_formatted/v0/passiv.netcdf"
N_PV_SYSTEMS_PER_EXAMPLE = 8

NWP_ZARR_PATH = (
    "/media/jack/wd_18tb/data/ocf/NWP/UK_Met_Office/UKV/zarr/UKV_intermediate_version_3.zarr"
)


def get_time_period(start_date, end_date):
    time_periods = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date, freq="30T"))
    time_periods["start_dt"] = time_periods.index
    time_periods["end_dt"] = time_periods.index + timedelta(minutes=30)

    return time_periods


def _get_nwp_data_source(
    zarr_path=NWP_ZARR_PATH,
    height_in_pixels=4,
    width_in_pixels=4,
    history_duration=datetime.timedelta(hours=1),
    forecast_duration=datetime.timedelta(hours=8),
    start_date="2020-01-01",
    end_date="2020-12-31 12:59",
) -> RawNWPDataSource:
    time_periods = get_time_period(start_date, end_date)

    return RawNWPDataSource(
        zarr_path=zarr_path,
        roi_height_pixels=height_in_pixels,
        roi_width_pixels=width_in_pixels,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        time_periods=time_periods,
        y_coarsen=16,
        x_coarsen=16,
        channels=["dswrf", "t", "si10", "prate"],
    )


@pytest.fixture(scope="session")
def nwp_data_source() -> RawNWPDataSource:
    return _get_nwp_data_source()


def _get_sat_data_source(
    zarr_path=(
        "/media/jack/wd_18tb/data/ocf/satellite/v3/eumetsat_seviri_hrv_uk.zarr"
        if USE_LOCAL_FILES
        else "gs://public-datasets-eumetsat-solar-forecasting/satellite/EUMETSAT/SEVIRI_RSS/v3/"
        "eumetsat_seviri_hrv_uk.zarr"
    ),
    height_in_pixels=SAT_HEIGHT_IN_PIXELS,
    width_in_pixels=SAT_WIDTH_IN_PIXELS,
    history_duration=datetime.timedelta(hours=1),
    forecast_duration=datetime.timedelta(hours=2),
    start_date="2020-01-01",
    end_date="2020-12-31 12:59",
) -> RawSatelliteDataSource:

    time_periods = get_time_period(start_date, end_date)

    return RawSatelliteDataSource(
        zarr_path=zarr_path,
        roi_height_pixels=height_in_pixels,
        roi_width_pixels=width_in_pixels,
        history_duration=history_duration,
        forecast_duration=forecast_duration,
        time_periods=time_periods
        # start_date=start_date,
        # end_date=end_date,
    )


@pytest.fixture(scope="session")
def get_sat_data_source():
    return _get_sat_data_source


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
    sat_data_source.per_worker_init(worker_id=1, seed=0)
    return sat_data_source


@pytest.fixture(scope="session")
def sat_data_loaded() -> RawSatelliteDataSource:
    # Don't use `sat_data_source` fixture, because that will run `per_worker_init`
    # on the "unopened" `sat_data_source` fixture!
    sat_data_source = _get_sat_data_source()
    sat_data_source.per_worker_init(worker_id=1, seed=0)

    periods = sat_data_source.get_contiguous_t0_time_periods()
    periods = periods.iloc[:3]
    sat_data_source.load_subset_into_ram(periods)

    return sat_data_source


@pytest.fixture(scope="session")
def pv_data_source() -> RawPVDataSource:

    time_periods = get_time_period("2020-01-01", "2020-01-03")

    pv = RawPVDataSource(
        pv_power_filename=PV_POWER_FILENAME,
        pv_metadata_filename=PV_METADATA_FILENAME,
        time_periods=time_periods,
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=2),
        roi_height_meters=64_000,
        roi_width_meters=64_000,
        n_pv_systems_per_example=N_PV_SYSTEMS_PER_EXAMPLE,
    )
    pv.per_worker_init(worker_id=0, seed=0)
    return pv


@pytest.fixture(scope="session")
def gsp_data_source() -> RawGSPDataSource:  # noqa: D103

    time_periods = get_time_period("2020-01-01", "2020-01-03")

    gsp = RawGSPDataSource(
        gsp_pv_power_zarr_path="~/data/PV/GSP/v3/pv_gsp.zarr",
        gsp_id_to_region_id_filename="~/data/PV/GSP/eso_metadata.csv",
        sheffield_solar_region_path="~/data/PV/GSP/gsp_shape",
        time_periods=time_periods,
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=2),
    )
    gsp.per_worker_init(worker_id=0, seed=0)
    return gsp


@pytest.fixture(scope="session")
def raw_dataset_with_sat_only(sat_data_source: RawSatelliteDataSource) -> RawDataset:
    return RawDataset(
        # deepcopy sat_data_source so we don't affect the fixture.
        data_source_combos=dict(sat_only=(deepcopy(sat_data_source),)),
        min_duration_to_load_per_epoch=datetime.timedelta(hours=20),
        n_batches_per_epoch=16,
        n_examples_per_batch=N_EXAMPLES_PER_BATCH,
    )


@pytest.fixture(scope="session")
def raw_dataset_with_sat_only_and_gsp_pv_sat(
    sat_data_source: RawSatelliteDataSource,
    pv_data_source: RawPVDataSource,
    gsp_data_source: RawGSPDataSource,
) -> RawDataset:
    return RawDataset(
        data_source_combos=dict(
            # deepcopy sat_data_source so we don't affect the fixture.
            sat_only=(deepcopy(sat_data_source),),
            gsp_pv_sat=(gsp_data_source, pv_data_source, deepcopy(sat_data_source)),
        ),
        min_duration_to_load_per_epoch=datetime.timedelta(hours=20),
        n_batches_per_epoch=16,
        n_examples_per_batch=N_EXAMPLES_PER_BATCH,
    )
