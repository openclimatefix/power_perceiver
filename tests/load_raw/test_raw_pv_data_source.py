from copy import copy

import numpy as np
import pytest
import xarray as xr
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.consts import Location
from power_perceiver.load_raw.data_sources.raw_pv_data_source import (
    RawPVDataSource,
    _load_pv_metadata,
    _load_pv_power_watts_and_capacity_wp,
)

# TODO: Use public data :)
PV_METADATA_FILENAME = "~/data/PV/Passiv/ocf_formatted/v0/system_metadata_OCF_ONLY.csv"
PV_POWER_FILENAME = "~/data/PV/Passiv/ocf_formatted/v0/passiv.netcdf"


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_load_pv_metadata():  # noqa: D103
    pv_metadata = _load_pv_metadata(PV_METADATA_FILENAME)
    assert len(pv_metadata) > 1000


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_load_pv_power_watts_and_capacity_wp():  # noqa: D103
    pv_power_watts, pv_capacity_wp, pv_system_row_number = _load_pv_power_watts_and_capacity_wp(
        PV_POWER_FILENAME, start_date="2020-01-01", end_date="2020-01-03"
    )
    assert len(pv_power_watts) == 863
    pv_system_ids = pv_power_watts.columns
    assert len(pv_system_ids) == 955
    assert np.array_equal(pv_capacity_wp.index, pv_system_ids)
    assert np.array_equal(pv_system_row_number.index, pv_system_ids)
    assert not pv_system_row_number.duplicated().any()
    assert not pv_system_ids.duplicated().any()
    assert np.isfinite(pv_system_row_number).all()
    assert np.isfinite(pv_system_ids).all()
    assert np.isfinite(pv_capacity_wp).all()
    assert np.all([col_dtype.type == np.float32 for col_dtype in pv_power_watts.dtypes.values])
    assert pv_capacity_wp.dtype.type == np.float32
    assert pv_system_ids.dtype.type == np.int64
    assert pv_system_row_number.dtype.type == np.float32


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_init(pv_data_source: RawPVDataSource):  # noqa: D103
    pv_power_normalised = pv_data_source._data_in_ram / pv_data_source._data_in_ram.capacity_wp
    assert pv_power_normalised.max().max() <= 1
    assert pv_power_normalised.dtype == np.float32


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
@pytest.mark.parametrize("n_pv_systems_per_example", [4, 7, 8, 16])
def test_get_spatial_slice(
    pv_data_source: RawPVDataSource, n_pv_systems_per_example: int
):  # noqa: D103
    pv_data_source = copy(pv_data_source)  # Don't modify the common pv_data_source.
    pv_data_source.n_pv_systems_per_example = n_pv_systems_per_example
    xr_data = pv_data_source._data_in_ram
    pv_system = xr_data.isel(pv_system_id=100)
    location = Location(x=pv_system.x_osgb.values, y=pv_system.y_osgb.values)
    spatial_slice = pv_data_source._get_spatial_slice(
        xr_data=xr_data,
        center_osgb=location,
    )
    assert len(spatial_slice.pv_system_id) == n_pv_systems_per_example
    assert len(spatial_slice.time_utc) == len(xr_data.time_utc)
    N_PV_SYSTEMS_AVAILABLE_IN_THIS_EXAMPLE = 7
    if n_pv_systems_per_example <= N_PV_SYSTEMS_AVAILABLE_IN_THIS_EXAMPLE:
        # There are 7 PV systems available. So assert there are no duplicates
        assert len(np.unique(spatial_slice.pv_system_id)) == len(spatial_slice.pv_system_id)
    else:
        assert len(np.unique(spatial_slice.pv_system_id)) == N_PV_SYSTEMS_AVAILABLE_IN_THIS_EXAMPLE


def _get_example(pv_data_source: RawPVDataSource) -> xr.DataArray:  # noqa: D103
    # Get valid location and t0_datetime for example:
    xr_data = pv_data_source._data_in_ram
    pv_system = xr_data.isel(pv_system_id=100)
    location = Location(x=pv_system.x_osgb.values, y=pv_system.y_osgb.values)
    contig_t0_periods = pv_data_source.get_contiguous_t0_time_periods()
    t0_dt = contig_t0_periods.iloc[0]["start_dt"]
    return pv_data_source.get_example(t0_datetime_utc=t0_dt, center_osgb=location)


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_get_example_and_empty_example(pv_data_source: RawPVDataSource) -> None:  # noqa: D103
    example = _get_example(pv_data_source)
    assert example.shape == pv_data_source.empty_example.shape
    assert example.dtype == np.float32


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_to_numpy(pv_data_source: RawPVDataSource) -> None:  # noqa: D103
    xr_example = _get_example(pv_data_source)
    np_example = RawPVDataSource.to_numpy(xr_example)
    assert np_example[BatchKey.pv].dtype == np.float32
    for batch_key, array in np_example.items():
        assert np.isfinite(array).all(), f"{batch_key=} has non-finite values!"
