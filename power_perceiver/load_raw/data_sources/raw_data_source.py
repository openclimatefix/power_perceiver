import datetime
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Callable, Iterable, Optional, Union

import pandas as pd
import xarray as xr

from power_perceiver.consts import Location
from power_perceiver.time import get_contiguous_time_periods
from power_perceiver.utils import check_path_exists


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
        self._check_input_paths_exist()
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
        self.open()

    def get_location_osgb(self) -> Location:
        """Randomly select a valid geographical location for one example.

        Must be overridden if this DataSource is to be used to select geographical locations
        for examples.

        Returns:  Location(x_center_osgb, y_center_osgb)
        """
        raise NotImplementedError()

    def get_example(self, t0_datetime_utc: datetime.datetime, center_osgb: Location) -> xr.Dataset:
        """Can be overridden by child classes.

        The returned Dataset must not include an `example` dimension.
        """
        xr_dataset = self.data_in_ram
        xr_dataset = self._get_time_slice(xr_dataset, t0_datetime_utc=t0_datetime_utc)
        xr_dataset = self._get_spatial_slice(xr_dataset, center_osgb=center_osgb)
        xr_dataset = self._transform(xr_dataset)
        return xr_dataset

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
    sample_period_duration: datetime.timedelta
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
        return pd.DatetimeIndex(self.data.time_utc)

    def _get_time_slice(
        self, xr_dataset: xr.Dataset, t0_datetime_utc: datetime.datetime
    ) -> xr.Dataset:
        """Select a timeslice from `xr_dataset`.

        The returned Dataset does not include an `example` dimension.
        """
        start_dt = self._get_start_dt(t0_datetime_utc)
        end_dt = self._get_end_dt(t0_datetime_utc)
        return xr_dataset.sel(time_utc=slice(start_dt, end_dt))

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
        # TODO: Maybe use the new contiguous time code from power_perceiver.
        return get_contiguous_time_periods(
            datetimes=self.datetime_index,
            min_seq_length=self.total_seq_length,
            max_gap_duration=self.sample_period_duration,
        )

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

    def __post_init__(self):
        assert self.height_in_pixels > 0, f"{self.height_in_pixels=} must be > 0!"
        assert self.width_in_pixels > 0, f"{self.width_in_pixels=} must be > 0!"
        assert (self.height_in_pixels % 2) == 0, f"{self.height_in_pixels=} must be divisible by 2!"
        assert (self.width_in_pixels % 2) == 0, f"{self.width_in_pixels=} must be divisible by 2!"

    def _get_spatial_slice(self, xr_dataset: xr.Dataset, center_osgb: Location) -> xr.Dataset:
        """Slice `xr_dataset` to produce a region of interest, centered on `center_osgb`.

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
        bottom_idx = center_idx.y - half_height
        top_idx = center_idx.y + half_height

        # Sanity check!
        assert left_idx >= 0, f"{left_idx=} must be >= 0!"
        assert bottom_idx >= 0, f"{bottom_idx=} must be >= 0!"
        assert right_idx <= len(xr_dataset.x), f"{right_idx=} must be <= {len(xr_dataset.x)=}"
        assert top_idx <= len(xr_dataset.y), f"{top_idx=} must be <= {len(xr_dataset.y)=}"

        xr_dataset = xr_dataset.isel(x=slice(left_idx, right_idx), y=slice(bottom_idx, top_idx))

        # Sanity check:
        assert len(xr_dataset.x) == self.width_in_pixels
        assert len(xr_dataset.y) == self.height_in_pixels

        return xr_dataset

    def _get_idx_of_pixel_at_center_of_roi(
        self, xr_dataset: xr.Dataset, center_osgb: Location
    ) -> Location:
        """Return x and y index location of pixel at center of region of interest."""
        raise NotImplementedError("Must be implemented by child class!")


@dataclass(kw_only=True)
class ZarrDatasource:
    """Abstract base class for Zarr stores."""

    zarr_path: Union[Path, str]

    def check_input_paths_exist(self) -> None:
        """Check input paths exist.  If not, raise a FileNotFoundError."""
        check_path_exists(self.zarr_path)
