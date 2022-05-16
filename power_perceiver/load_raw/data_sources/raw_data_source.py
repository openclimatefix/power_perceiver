import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
import xarray as xr

from power_perceiver.consts import Location
from power_perceiver.time import get_contiguous_time_periods
from power_perceiver.utils import check_path_exists

_log = logging.getLogger(__name__)


# `kw_only` allows a base class to have fields with default values,
# and it's still OK for child classes to have fields *without* defaults. See:
# https://stackoverflow.com/a/69822584/732596
@dataclass(kw_only=True)
class RawDataSource:
    """Abstract base class for loading data directly from the raw or intermediate files.

    Attributes:
        data_in_ram: xr.Dataset. Uses standard names for dimensions and coordinates. If the data
            does not conform to these standard names, then the subclass must change the names
            as soon as the Dataset is opened. The standard names are:
                Dimension names: time, channel, y, x
                Coordinate names: time_utc, y_osbg, x_osgb
        data_on_disk: xr.Dataset.
            If this DataSource loads a subset of the full dataset at the start of each epoch,
            then `data_in_ram` holds the in-memory data, and `data_on_disk` holds
            the un-loaded `xr.Dataset`. If this DataSource can load the entire dataset into RAM,
            then `data_on_disk` will be `None`.
    """

    transforms: Optional[Iterable[Callable]] = None

    def __post_init__(self):  # noqa: D105
        self._data_in_ram = None
        self._data_on_disk = None

    @property
    def data_in_ram(self):  # noqa: D102
        if self._data_in_ram is None:
            raise RuntimeError("Please load the data into RAM before accessing `data_in_ram`!")
        return self._data_in_ram

    @property
    def data_on_disk(self):  # noqa: D102
        if self._data_on_disk is None:
            raise RuntimeError("Please open the dataset before accessing `data_on_disk`!")
        return self._data_on_disk

    def per_worker_init(self, worker_id: int) -> None:  # noqa: D102
        self.worker_id = worker_id
        # Each worker must have a different seed for its random number generator.
        # Otherwise all the workers will output exactly the same data!
        seed = torch.initial_seed()
        _log.info(f"{worker_id=} has random number generator {seed=:,d}")
        self.rng = np.random.default_rng(seed=seed)
        self.open()

    def get_example(self, t0_datetime_utc: datetime.datetime, center_osgb: Location) -> xr.Dataset:
        """Can be overridden by child classes.

        The returned Dataset must not include an `example` dimension.
        """
        xr_dataset = self.data_in_ram
        xr_dataset = self._get_time_slice(xr_dataset, t0_datetime_utc=t0_datetime_utc)
        xr_dataset = self._get_spatial_slice(xr_dataset, center_osgb=center_osgb)
        xr_dataset = self._post_process(xr_dataset)
        xr_dataset = self._transform(xr_dataset)
        return xr_dataset

    def get_empty_example(self) -> xr.Dataset:
        """Must be overridden by child classes.

        The returned Dataset must not include an `example` dimension.
        """
        raise NotImplementedError()

    def open(self):
        """Open the data source, if necessary.

        Called from each worker process. Useful for data sources where the
        underlying data source cannot be forked (like Zarr).

        Data sources which can be forked safely should call open() from __init__().
        """
        pass

    def _get_time_slice(
        self, xr_dataset: xr.Dataset, t0_datetime_utc: datetime.datetime
    ) -> xr.Dataset:
        """Can be overridden, usually by TimeseriesDataSource.

        The returned Dataset does not include an `example` dimension.
        """
        return xr_dataset

    def _get_spatial_slice(self, xr_dataset: xr.Dataset, center_osgb: Location) -> xr.Dataset:
        """Can be overridden, usually by SpatialDataSource.

        The returned Dataset does not include an `example` dimension.
        """
        return xr_dataset

    def _post_process(self, xr_dataset: xr.Dataset) -> xr.Dataset:
        """Can be overridden. This is where normalisation should happen.

        The returned Dataset does not include an `example` dimension.
        """
        return xr_dataset

    def _check_input_paths_exist(self) -> None:
        """Check any input paths exist.  Raise FileNotFoundError if not.

        Must be overridden by child classes.
        """
        raise NotImplementedError()

    def _transform(self, xr_dataset: xr.Dataset) -> xr.Dataset:
        if self.transforms:
            for transform in self.transforms:
                xr_dataset = transform(xr_dataset)
        return xr_dataset


@dataclass(kw_only=True)
class TimeseriesDataSource:
    """Abstract base class mixin for timeseries data sources.

    Init args:
        history_duration: Total duration of the history *including* t0.
            (This is a different definition to the definition used in `nowcasting_dataset`,
            which *excludes* t0 from `history_minutes`.) `history_duration` is the duration
            between the first timestep and t0. We consider t0 to be part of the history.
            `history_duration` can be zero. If `history_duration` and `forecast_duration` are both
            zero then there will still be a single timestep at t0.
        forecast_duration: Total duration of the forecast: dt_end - t0.
        sample_period_duration: The maximum legal `timedelta` between samples. Cannot be zero.
        start_date:
        end_date:
    """

    history_duration: datetime.timedelta
    forecast_duration: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime

    def __post_init__(self):  # noqa: D105
        # Sanity checks.
        assert self.sample_period_duration, "sample_period_duration cannot be zero length!"
        assert (
            self.start_date < self.end_date
        ), f"{self.start_date=} must be before {self.end_date=}"
        if self.history_duration:
            assert (
                self.history_duration >= self.sample_period_duration
            ), f"{self.history_duration=} must be zero or >= {self.sample_period_duration}"

        if self.forecast_duration:
            assert (
                self.forecast_duration >= self.sample_period_duration
            ), f"{self.forecast_duration=} must be zero or >= {self.sample_period_duration}"

    @property
    def total_seq_length(self) -> int:
        """Total number of timesteps per example, including t0."""
        return int(self.total_duration / self.sample_period_duration) + 1  # Plus 1 for t0.

    @property
    def total_duration(self) -> datetime.timedelta:
        """Total number of duration per example, including t0."""
        return self.history_duration + self.forecast_duration

    @property
    def datetime_index(self) -> pd.DatetimeIndex:
        """Return a pd.DatetimeIndex of all available datetimes.

        Child classes can override this to, for example, filter out any datetimes
        which don't make sense for this DataSource, e.g. remove nighttime.
        """
        return pd.DatetimeIndex(self.data_on_disk.time_utc)

    @property
    def sample_period_duration(self) -> datetime.timedelta:
        raise NotImplementedError("Must be overridden by child class!")

    @property
    def needs_to_load_subset_into_ram(self) -> bool:
        """Override in subclasses which need to load subset into RAM.

        If this returns True, then you must also implement `load_subset_into_ram`.
        """
        return False

    def load_subset_into_ram(self, subset_of_contiguous_time_periods: pd.DataFrame) -> None:
        """Override in DataSources which can only fit a subset of the dataset into RAM."""
        raise NotImplementedError()

    def get_contiguous_t0_time_periods(self) -> pd.DataFrame:
        """Get all time periods which contain valid t0 datetimes.

        `t0` is the datetime of the most recent observation.

        Returns:
          pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
        """
        contiguous_time_periods = self._get_contiguous_time_periods()
        contiguous_time_periods["start_dt"] += self.history_duration
        contiguous_time_periods["end_dt"] -= self.forecast_duration
        assert (contiguous_time_periods["start_dt"] <= contiguous_time_periods["end_dt"]).all()
        return contiguous_time_periods

    def _get_contiguous_time_periods(self) -> pd.DataFrame:
        """Get all the time periods for which this DataSource has contiguous data.

        Returns:
          pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
        """
        return get_contiguous_time_periods(
            datetimes=self.datetime_index,
            min_seq_length=self.total_seq_length,
            max_gap_duration=self.sample_period_duration,
        )

    def _get_time_slice(
        self, xr_dataset: xr.Dataset, t0_datetime_utc: datetime.datetime
    ) -> xr.Dataset:
        """Select a timeslice from `xr_dataset`.

        The returned Dataset does not include an `example` dimension.
        """
        start_dt = self._get_start_dt(t0_datetime_utc)
        end_dt = self._get_end_dt(t0_datetime_utc)
        return xr_dataset.sel(time_utc=slice(start_dt, end_dt))

    def _get_start_dt(self, t0_datetime_utc: datetime.datetime) -> datetime.datetime:
        return t0_datetime_utc - self.history_duration

    def _get_end_dt(self, t0_datetime_utc: datetime.datetime) -> datetime.datetime:
        return t0_datetime_utc + self.forecast_duration


@dataclass(kw_only=True)
class SpatialDataSource:
    """Abstract base class for image Data source.

    Args:
        height_in_pixels: Must be divisible by 2.
        width_in_pixels: Must be divisible by 2.
    """

    height_in_pixels: int
    width_in_pixels: int

    # Attributes which are intended to be set for the whole class.
    _y_dim_name: str = "y"
    _x_dim_name: str = "x"

    def __post_init__(self):
        assert self.height_in_pixels > 0, f"{self.height_in_pixels=} must be > 0!"
        assert self.width_in_pixels > 0, f"{self.width_in_pixels=} must be > 0!"
        assert (self.height_in_pixels % 2) == 0, f"{self.height_in_pixels=} must be divisible by 2!"
        assert (self.width_in_pixels % 2) == 0, f"{self.width_in_pixels=} must be divisible by 2!"

    def get_location_osgb_for_example(self) -> Location:
        """Randomly select a valid geographical location for one example.

        Must be overridden if this DataSource is to be used to select geographical locations
        for examples.

        Returns:  Location(x_center_osgb, y_center_osgb)
        """
        raise NotImplementedError()

    def _get_spatial_slice(self, xr_dataset: xr.Dataset, center_osgb: Location) -> xr.Dataset:
        """Slice `xr_dataset` to produce a region of interest, centered on `center_osgb`.

        Assume the image data starts top-left.

        The returned Dataset does not include an `example` dimension.
        """
        # Find pixel index at `center_osgb`:
        center_idx = self._get_idx_of_pixel_at_center_of_roi(
            xr_dataset=xr_dataset, center_osgb=center_osgb
        )

        # Compute the index for left and right:
        half_height = self.height_in_pixels // 2
        half_width = self.width_in_pixels // 2

        left_idx = center_idx.x - half_width
        right_idx = center_idx.x + half_width
        top_idx = center_idx.y - half_height
        bottom_idx = center_idx.y + half_height

        # Sanity check!
        assert left_idx >= 0, f"{left_idx=} must be >= 0!"
        data_width_pixels = len(xr_dataset[self._x_dim_name])
        assert right_idx <= data_width_pixels, f"{right_idx=} must be <= {data_width_pixels=}"
        assert top_idx >= 0, f"{top_idx=} must be >= 0!"
        data_height_pixels = len(xr_dataset[self._y_dim_name])
        assert bottom_idx <= data_height_pixels, f"{bottom_idx=} must be <= {data_height_pixels=}"

        selected = xr_dataset.isel(
            x_geostationary=slice(left_idx, right_idx), y_geostationary=slice(top_idx, bottom_idx)
        )

        # Sanity check:
        assert len(selected[self._x_dim_name]) == self.width_in_pixels
        assert len(selected[self._y_dim_name]) == self.height_in_pixels

        return selected

    def _get_idx_of_pixel_at_center_of_roi(
        self, xr_dataset: xr.Dataset, center_osgb: Location
    ) -> Location:
        """Return x and y index location of pixel at center of region of interest."""
        raise NotImplementedError("Must be implemented by child class!")


@dataclass(kw_only=True)
class ZarrDataSource:
    """Abstract base class for Zarr stores."""

    zarr_path: Union[Path, str]

    def __post_init__(self):
        check_path_exists(self.zarr_path)
