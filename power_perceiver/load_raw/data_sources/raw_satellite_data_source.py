import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import dask
import numpy as np
import pandas as pd
import pyproj
import pyresample
import xarray as xr

from power_perceiver.consts import BatchKey, Location
from power_perceiver.geospatial import OSGB
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    SpatialDataSource,
    TimeseriesDataSource,
    ZarrDatasource,
)
from power_perceiver.time import (
    date_summary_str,
    select_data_in_daylight,
    select_timesteps_in_contiguous_periods,
)
from power_perceiver.utils import datetime64_to_float, is_sorted

_log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RawSatelliteDataSource(
    RawDataSource, TimeseriesDataSource, SpatialDataSource, ZarrDatasource
):
    """Load satellite data directly from the satellite Zarr store."""

    @property
    def sample_period_duration(self) -> datetime.timedelta:  # noqa: D102
        return datetime.timedelta(minutes=5)

    @property
    def datetime_index(self) -> pd.DatetimeIndex:
        raise NotImplementedError()  # Still TODO! Filter out nighttime.

    @property
    def needs_to_load_subset_into_ram(self) -> bool:  # noqa: D102
        return True

    def load_subset_into_ram(self, subset_of_contiguous_time_periods: pd.DataFrame) -> None:
        """Override in DataSources which can only fit a subset of the dataset into RAM."""
        raise NotImplementedError()  # TODO!

    def open(self) -> None:
        """
        Open Satellite data

        We don't want to open_sat_data in __init__.
        If we did that, then we couldn't copy SatelliteDataSource
        instances into separate processes.  Instead,
        call open() _after_ creating separate processes.
        """
        self._data_on_disk = open_sat_data(zarr_path=self.zarr_path)

        self._load_geostationary_area_definition_and_transform()

        # Check the x and y coords are ascending. If they are not then searchsorted won't work!
        assert is_sorted(self.data_on_disk.x_geostationary)
        assert is_sorted(self.data_on_disk.y_geostationary)

        # Sub-select data:
        _log.info("Before any selection, " + date_summary_str(self._data_on_disk))

        # Select only the timesteps we want:
        self._data_on_disk = self._data_on_disk.sel(time_utc=slice(self.start_date, self.end_date))
        self._data_on_disk = select_data_in_daylight(self._data_on_disk)
        _log.info("After filtering, " + date_summary_str(self._data_on_disk))

    @staticmethod
    def to_numpy_batch(xr_data: xr.DataArray) -> NumpyBatch:
        example: NumpyBatch = {}
        # Insert a "channels" dimension:
        example[BatchKey.hrvsatellite] = xr_data.expand_dims(dim="channel", axis=2).values.copy()

        # Insert example dimensions:
        example[BatchKey.hrvsatellite_time_utc] = datetime64_to_float(
            xr_data["time"].expand_dims(dim="example", axis=0).values.copy()
        )
        for batch_key, dataset_key in (
            (BatchKey.hrvsatellite_y_osgb, "y_osgb"),
            (BatchKey.hrvsatellite_x_osgb, "x_osgb"),
            (BatchKey.hrvsatellite_y_geostationary, "y"),
            (BatchKey.hrvsatellite_x_geostationary, "x"),
        ):
            # HRVSatellite coords are already float32.
            example[batch_key] = (
                xr_data[dataset_key].expand_dims(dim="example", axis=0).values.copy()
            )
        return example

    def _load_geostationary_area_definition_and_transform(self) -> None:
        area_definition_yaml = self._data_on_disk.attrs["area"]
        geostationary_area_definition = pyresample.area_config.load_area_from_string(
            area_definition_yaml
        )
        geostationary_crs = geostationary_area_definition.crs
        self._osgb_to_geostationary = pyproj.Transformer.from_crs(
            crs_from=OSGB, crs_to=geostationary_crs
        ).transform

    def _get_idx_of_pixel_at_center_of_roi(
        self, xr_dataset: xr.Dataset, center_osgb: Location
    ) -> Location:
        """Return x and y index location of pixel at center of region of interest."""
        center_geostationary = self._osgb_to_geostationary(center_osgb)
        # Get the index into x and y nearest to x_center_geostationary and y_center_geostationary:
        x_index_at_center = (
            np.searchsorted(xr_dataset.x_geostationary.values, center_geostationary.x) - 1
        )
        y_index_at_center = (
            np.searchsorted(xr_dataset.y_geostationary.values, center_geostationary.y) - 1
        )
        return Location(x=x_index_at_center, y=y_index_at_center)


def open_sat_data(zarr_path: Union[Path, str]) -> xr.DataArray:
    """Lazily opens the Zarr store.

    Rounds the 'time' coordinates, so the timestamps are at 00, 05, ..., 55 past the hour.

    Args:
      zarr_path: Cloud URL or local path pattern.  If GCP URL, must start with 'gs://'
    """
    _log.debug("Opening satellite data: %s", zarr_path)

    # Silence the warning about large chunks.
    # Alternatively, we could set this to True, but that slows down loading a Satellite batch
    # from 8 seconds to 50 seconds!
    dask.config.set({"array.slicing.split_large_chunks": False})

    # Open the data
    dataset = xr.open_dataset(zarr_path, engine="zarr", chunks="auto")

    # Rename
    # These renamings will no longer be necessary when the Zarr uses the 'correct' names,
    # see https://github.com/openclimatefix/Satip/issues/66
    if "x" in dataset:
        dataset = dataset.rename({"x": "x_geostationary", "y": "y_geostationary"})
    if "variable" in dataset:
        dataset = dataset.rename({"variable": "channels"})
    elif "channels" not in dataset:
        # This is HRV version 3, which doesn't have a channels dim.  So add one.
        dataset = dataset.expand_dims(dim={"channels": ["HRV"]}, axis=-1)

    data_array = dataset["data"]
    if "stacked_eumetsat_data" == data_array.name:
        data_array.name = "data"
    del dataset

    # Flip coordinates to top-left first
    data_array = data_array.reindex(x_geostationary=data_array.x_geostationary[::-1])

    # Round datetimes to the nearest 5 minutes.
    # (Satellite datetimes can sometimes be 04, 09, minutes past the hour, or other slight offsets.
    # These slight offsets will break downstream code, which expects satellite data to be at
    # exactly 5 minutes past the hour).
    datetime_index = pd.DatetimeIndex(data_array.time)
    datetime_index = datetime_index.round("5T")
    data_array.time = datetime_index

    # Sanity check!
    assert datetime_index.is_unique
    assert datetime_index.is_monotonic_increasing

    return data_array
