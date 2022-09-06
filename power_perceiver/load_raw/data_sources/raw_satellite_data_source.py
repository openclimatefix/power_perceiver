import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Union

import dask
import numpy as np
import pandas as pd
import pyproj
import pyresample
import xarray as xr
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.consts import Location
from power_perceiver.geospatial import OSGB36
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_prepared_batches.data_sources.satellite import SAT_MEAN, SAT_STD
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    SpatialDataSource,
    TimeseriesDataSource,
    ZarrDataSource,
)
from power_perceiver.time import date_summary_str, select_data_in_daylight
from power_perceiver.utils import datetime64_to_float, is_sorted, select_time_periods

_log = logging.getLogger(__name__)


@dataclass(kw_only=True)
class RawSatelliteDataSource(
    # Surprisingly, Python's class hierarchy is defined right-to-left.
    # So the base class must go on the right.
    ZarrDataSource,
    TimeseriesDataSource,
    SpatialDataSource,
    RawDataSource,
):
    """Load satellite data directly from the intermediate satellite Zarr store."""

    _y_dim_name: ClassVar[str] = "y_geostationary"
    _x_dim_name: ClassVar[str] = "x_geostationary"
    # For now, let's assume the satellite imagery is always 5-minutely.
    # Later (WP3?), we'll want to experiment with lower temporal resolution satellite imagery.
    sample_period_duration: ClassVar[datetime.timedelta] = datetime.timedelta(minutes=5)
    needs_to_load_subset_into_ram: ClassVar[bool] = True

    def __post_init__(self):  # noqa: D105
        RawDataSource.__post_init__(self)
        SpatialDataSource.__post_init__(self)
        TimeseriesDataSource.__post_init__(self)
        ZarrDataSource.__post_init__(self)

    def per_worker_init(self, *args, **kwargs) -> None:  # noqa: D102
        super().per_worker_init(*args, **kwargs)
        self.open()

    def open(self) -> None:
        """
        Lazily open Satellite data (this only loads metadata. It doesn't load all the data!)

        We don't want to call `open_sat_data` in __init__.
        If we did that, then we couldn't copy RawSatelliteDataSource
        instances into separate processes.  Instead,
        call open() _after_ creating separate processes.
        """
        self._data_on_disk = open_sat_data(zarr_path=self.zarr_path)

        self._load_geostationary_area_definition_and_transform()

        # Check the x and y coords are sorted. If they are not then searchsorted won't work!
        assert is_sorted(self.data_on_disk[self._y_dim_name][::-1])
        assert is_sorted(self.data_on_disk[self._x_dim_name])

        # Sub-select data:
        _log.info("Before any selection: " + date_summary_str(self.data_on_disk.time_utc))

        # Select only the timesteps we want:
        self._data_on_disk = select_time_periods(
            xr_data=self.data_on_disk,
            time_periods=self.time_periods,
        )
        self._data_on_disk = select_data_in_daylight(self._data_on_disk)
        _log.info("After filtering: " + date_summary_str(self.data_on_disk.time_utc))

    def _post_process(self, xr_data: xr.DataArray) -> xr.DataArray:
        # hrvsatellite is int16 on disk but, upstream of this method, we convert to uint8 to save
        # RAM. We convert to uint8 by dividing by 4 (to get from pixel values in the range
        # [0, 1023] to the range [0, 255]). So `xr_data` will be uint8 when it gets to this method.
        # But `SAT_MEAN` and `SAT_STD` were computed on the original satellite data,
        # which is in the range [0, 1023] (i.e. 4 times greater than the uint8 range of [0, 255])
        # We divide the mean and std by 4 because the `xr_data` has also been divided by 4.
        xr_data = xr_data.astype(np.float32)
        xr_data = xr_data - (SAT_MEAN["HRV"] / 4)
        xr_data = xr_data / (SAT_STD["HRV"] / 4)
        return xr_data

    @staticmethod
    def to_numpy(xr_data: xr.DataArray) -> NumpyBatch:
        """Convert xarray to numpy batch.

        But note that this is actually just returns *one* example (not a whole batch!)
        """
        example: NumpyBatch = {}

        example[BatchKey.hrvsatellite_actual] = xr_data.values
        example[BatchKey.hrvsatellite_t0_idx] = xr_data.attrs["t0_idx"]
        example[BatchKey.hrvsatellite_time_utc] = datetime64_to_float(xr_data["time_utc"].values)
        for batch_key, dataset_key in (
            (BatchKey.hrvsatellite_y_osgb, "y_osgb"),
            (BatchKey.hrvsatellite_x_osgb, "x_osgb"),
            (BatchKey.hrvsatellite_y_geostationary, RawSatelliteDataSource._y_dim_name),
            (BatchKey.hrvsatellite_x_geostationary, RawSatelliteDataSource._x_dim_name),
        ):
            # HRVSatellite coords are already float32.
            example[batch_key] = xr_data[dataset_key].values

        return example

    def _load_geostationary_area_definition_and_transform(self) -> None:
        area_definition_yaml = self.data_on_disk.attrs["area"]
        geostationary_area_definition = pyresample.area_config.load_area_from_string(
            area_definition_yaml
        )
        geostationary_crs = geostationary_area_definition.crs
        self._osgb_to_geostationary = pyproj.Transformer.from_crs(
            crs_from=OSGB36, crs_to=geostationary_crs
        ).transform

    def _get_idx_of_pixel_at_center_of_roi(
        self, xr_data: xr.DataArray, center_osgb: Location
    ) -> Location:
        """Return x and y index location of pixel at center of region of interest."""
        center_geostationary_tuple = self._osgb_to_geostationary(xx=center_osgb.x, yy=center_osgb.y)
        center_geostationary = Location(
            x=center_geostationary_tuple[0], y=center_geostationary_tuple[1]
        )

        # Get the index into x and y nearest to x_center_geostationary and y_center_geostationary:
        x_index_at_center = (
            np.searchsorted(xr_data[self._x_dim_name].values, center_geostationary.x) - 1
        )
        # y_geostationary is in descending order:
        y_index_at_center = len(xr_data[self._y_dim_name]) - (
            np.searchsorted(xr_data[self._y_dim_name].values[::-1], center_geostationary.y) - 1
        )
        return Location(x=x_index_at_center, y=y_index_at_center)


def open_sat_data(
    zarr_path: Union[Path, str],
    convert_to_uint8: bool = True,
) -> xr.DataArray:
    """Lazily opens the Zarr store.

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

    # Flip coordinates to top-left first
    dataset = dataset.reindex(y=dataset.y[::-1])
    dataset = dataset.reindex(x=dataset.x[::-1])

    # Rename
    # These renamings will no longer be necessary when the Zarr uses the 'correct' names,
    # see https://github.com/openclimatefix/Satip/issues/66
    if "variable" in dataset:
        dataset = dataset.rename({"variable": "channel"})
    elif "channel" not in dataset:
        # This is HRV version 3, which doesn't have a channels dim.  So add one.
        dataset = dataset.expand_dims(dim={"channel": ["HRV"]}, axis=1)

    # Rename coords to be more explicit about exactly what some coordinates hold:
    # Note that `rename` renames *both* the coordinates and dimensions, and keeps
    # the connection between the dims and coordinates, so we don't have to manually
    # use `data_array.set_index()`.
    dataset = dataset.rename(
        {
            "time": "time_utc",
            "y": "y_geostationary",
            "x": "x_geostationary",
        }
    )

    data_array = dataset["data"]
    del dataset

    # Ensure the y and x coords are in the right order (top-left first):
    assert data_array.y_geostationary[0] > data_array.y_geostationary[-1]
    assert data_array.x_geostationary[0] < data_array.x_geostationary[-1]
    assert data_array.y_osgb[0, 0] > data_array.y_osgb[-1, 0]
    assert data_array.x_osgb[0, 0] < data_array.x_osgb[0, -1]

    if convert_to_uint8:
        data_array = data_array.clip(min=0, max=1023)
        data_array.data = (data_array.astype(np.float32).data / 4.0).round().astype(np.uint8)

    # Sanity checks!
    assert data_array.dims == ("time_utc", "channel", "y_geostationary", "x_geostationary")
    datetime_index = pd.DatetimeIndex(data_array.time_utc)
    assert datetime_index.is_unique
    assert datetime_index.is_monotonic_increasing
    # Satellite datetimes can sometimes be 04, 09, minutes past the hour, or other slight offsets.
    # These slight offsets will break downstream code, which expects satellite data to be at
    # exactly 5 minutes past the hour.
    assert (datetime_index == datetime_index.round("5T")).all()

    return data_array
