import datetime
import logging
from dataclasses import dataclass

import dask
import numpy as np
import pandas as pd
import pyproj
import pyresample
import xarray as xr

from power_perceiver.consts import Location
from power_perceiver.geospatial import OSGB
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    SpatialDataSource,
    TimeseriesDataSource,
    ZarrDatasource,
)
from power_perceiver.utils import is_sorted

_log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RawSatelliteDataSource(
    RawDataSource, TimeseriesDataSource, SpatialDataSource, ZarrDatasource
):
    @property
    def sample_period_duration(self) -> datetime.timedelta:  # noqa: D102
        return datetime.timedelta(minutes=5)

    @property
    def datetime_index(self) -> pd.DatetimeIndex:
        raise NotImplementedError()  # Still TODO! Filter out nighttime.

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


def open_sat_data(zarr_path: str) -> xr.DataArray:
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
