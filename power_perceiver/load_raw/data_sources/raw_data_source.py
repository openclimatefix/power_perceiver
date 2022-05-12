import datetime
from dataclasses import dataclass
from numbers import Number
from typing import Callable, Iterable, Optional

import pandas as pd
import xarray as xr


# `kw_only` allows a base class to have fields with default values,
# and it's still OK for child classes to have fields *without* defaults. See:
# https://stackoverflow.com/a/69822584/732596
@dataclass(kw_only=True)
class RawDataSource:
    """Abstract base class for loading data directly from the raw or intermediate files."""

    transforms: Optional[Iterable[Callable]] = None

    def __post_init__(self):  # noqa: D105
        self._check_input_paths_exist()
        self._data = None

    @property
    def data(self):  # noqa: D102
        if self._data is None:
            raise RuntimeError("Please run `open()` before accessing data!")
        return self._data

    def get_location(self) -> tuple[Number, Number]:
        """Randomly select a valid geographical location for one example.

        Must be overridden if this DataSource is to be used to select geographical locations
        for examples.

        Returns:  x_center_osgb, y_center_osgb
        """
        raise NotImplementedError()

    def get_example(
        self,
        t0_datetime_utc: datetime.datetime,
        x_center_osgb: Number,
        y_center_osgb: Number,
    ) -> xr.Dataset:
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
    """Abstract base class mixin for timeseries data sources."""

    history_duration: datetime.timedelta
    forecast_duration: datetime.timedelta
    start_date: datetime.datetime
    end_date: datetime.datetime

    def get_contiguous_t0_time_periods(self) -> pd.DataFrame:
        """Get all time periods which contain valid t0 datetimes.

        `t0` is the datetime of the most recent observation.

        Returns:
          pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').

        Raises:
          NotImplementedError if this DataSource has no concept of a datetime index.
        """
        contiguous_time_periods = self.get_contiguous_time_periods()
        contiguous_time_periods["start_dt"] += self.history_duration
        contiguous_time_periods["end_dt"] -= self.forecast_duration
        assert (contiguous_time_periods["start_dt"] <= contiguous_time_periods["end_dt"]).all()
        return contiguous_time_periods

    def get_contiguous_time_periods(self) -> pd.DataFrame:
        """Get all the time periods for which this DataSource has contiguous data.

        Optionally filter out any time periods which don't make sense for this DataSource,
        e.g. remove nighttime.

        Returns:
          pd.DataFrame where each row represents a single time period.  The pd.DataFrame
          has two columns: `start_dt` and `end_dt` (where 'dt' is short for 'datetime').

        Raises:
          NotImplementedError if this DataSource has no concept of a datetime index.
        """
        datetimes = self.datetime_index()
        return nd_time.get_contiguous_time_periods(
            datetimes=datetimes,
            min_seq_length=self.total_seq_length,  # TODO: Use total seq duration?
            max_gap_duration=self.sample_period_duration,
        )

    def datetime_index(self) -> pd.DatetimeIndex:
        """Return a complete list of all available datetimes."""
        raise NotImplementedError(f"Datetime not implemented for class {self.__class__}")

    def _get_start_dt(self, t0_datetime_utc: datetime.datetime) -> datetime.datetime:
        return t0_datetime_utc - self.history_duration

    def _get_end_dt(self, t0_datetime_utc: datetime.datetime) -> datetime.datetime:
        return t0_datetime_utc + self.forecast_duration


@dataclass(kw_only=True)
class ImageDataSource:
    """Abstract base class for image Data source."""

    image_size_pixels_height: int
    image_size_pixels_width: int

    def _get_spatial_slice(
        self, x_center_osgb: Number, y_center_osgb: Number, xr_dataset: Optional[xr.Dataset] = None
    ) -> xr.Dataset:
        if xr_dataset is None:
            xr_dataset = self.data

        # TODO:
        # Find pixel index at x_center_osgb, y_center_osgb. Then
        # just add and subtract image_size_pixels_height etc.
        # and use `isel` to grab the data. This should work for
        # satellite, NWP, topo, etc.
