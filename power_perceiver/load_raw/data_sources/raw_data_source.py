import datetime
import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Iterable, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr

from power_perceiver.consts import Location
from power_perceiver.exceptions import NoPVSystemsInSlice
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
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
        data_in_ram: xr.DataArray. Uses standard names for dimensions and coordinates. If the data
            does not conform to these standard names, then the subclass must change the names
            as soon as the DataArray is opened. The standard names are:
                Dimension names: time_utc, channel
                Coordinate names: time_utc, y_osbg, x_osgb, y_geostationary, x_geostationary
        data_on_disk: xr.DataArray.
            If this DataSource loads a subset of the full DataArray at the start of each epoch,
            then `data_in_ram` holds the in-memory data, and `data_on_disk` holds
            the un-loaded `xr.DataArray`. If this DataSource can load the entire DataArray into RAM,
            then `data_on_disk` will be `None`.
    """

    transforms: Optional[Iterable[Callable]] = None
    needs_to_load_subset_into_ram: ClassVar[bool] = False

    def __post_init__(self):  # noqa: D105
        self._data_in_ram = None
        self._data_on_disk = None
        # For data sources that can load everything into RAM,
        # override `__post_init__` to load everything into RAM here.
        # Otherwise, override `per_worker_init` to load everything into RAM there.

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

    def per_worker_init(self, worker_id: int, seed: int) -> None:  # noqa: D102
        self.worker_id = worker_id
        # Each worker must have a different seed for its random number generator.
        # Otherwise all the workers will output exactly the same data!
        _log.info(f"{worker_id=} has random number generator {seed=:,d}")
        self.rng = np.random.default_rng(seed=seed)
        # Optionally, override `per_worker_init` in child class to call super().per_worker_init()
        # and to `open()` the data, if opening doesn't happen in `__post_init__().`

    def get_example(
        self, t0_datetime_utc: datetime.datetime, center_osgb: Location
    ) -> xr.DataArray:
        """Can be overridden by child classes.

        The returned Dataset must not include an `example` dimension.
        """
        using_empty_example = False
        try:
            xr_data = self._get_slice(t0_datetime_utc=t0_datetime_utc, center_osgb=center_osgb)
        except NoPVSystemsInSlice:
            xr_data = self.empty_example
            using_empty_example = True
        xr_data = self._post_process(xr_data)
        xr_data = self._set_attributes(xr_data)
        xr_data = self._transform(xr_data)

        if not using_empty_example:
            try:
                self.check_xarray_data(xr_data)
            except Exception as e:
                raise e.__class__(
                    f"Exception raised when checking xr data! {t0_datetime_utc=} {center_osgb=}"
                ) from e
        return xr_data

    def _get_slice(self, t0_datetime_utc: datetime.datetime, center_osgb: Location) -> xr.DataArray:
        """Can be overridden by child classes.

        The returned Dataset must not include an `example` dimension.
        """
        xr_data = self.data_in_ram
        xr_data = self._get_time_slice(xr_data, t0_datetime_utc=t0_datetime_utc)
        xr_data = self._get_spatial_slice(xr_data, center_osgb=center_osgb)
        return xr_data

    def check_xarray_data(self, xr_data: xr.DataArray):  # noqa: D102
        assert np.isfinite(xr_data).all(), "Some xr_data is non-finite!"

    def _set_attributes(self, xr_data: xr.DataArray) -> xr.DataArray:
        return xr_data

    @staticmethod
    def to_numpy(xr_data: xr.DataArray) -> NumpyBatch:
        """Convert xarray to numpy batch.

        But note that this is actually just returns *one* example (not a whole batch!)
        """
        raise NotImplementedError("Must be implemented by subclass!")

    def _get_time_slice(
        self, xr_data: xr.DataArray, t0_datetime_utc: datetime.datetime
    ) -> xr.DataArray:
        """Can be overridden, usually by TimeseriesDataSource.

        The returned Dataset does not include an `example` dimension.

        This method should sanity check that the example is the correct duration. But shouldn't
        check for NaNs (because subsequent processing might remove NaNs.
        `check_xarray_data` checks for NaNs.
        """
        return xr_data

    def _get_spatial_slice(self, xr_data: xr.DataArray, center_osgb: Location) -> xr.DataArray:
        """Can be overridden, usually by SpatialDataSource.

        The returned Dataset does not include an `example` dimension.

        This method should sanity check that the example is the correct spatial shape. But shouldn't
        check for NaNs (because subsequent processing might remove NaNs.
        `check_xarray_data` checks for NaNs.
        """
        return xr_data

    def _post_process(self, xr_data: xr.DataArray) -> xr.DataArray:
        """Can be overridden. This is where normalisation should happen.

        The returned Dataset does not include an `example` dimension.
        """
        return xr_data

    def _transform(self, xr_data: xr.DataArray) -> xr.DataArray:
        if self.transforms:
            for transform in self.transforms:
                xr_data = transform(xr_data)
        return xr_data


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
        time_periods: The time periods to consider. A pd.DataFrame with two columns:
            start_dt and end_dt.
    """

    history_duration: datetime.timedelta
    forecast_duration: datetime.timedelta
    time_periods: pd.DataFrame
    sample_period_duration: ClassVar[datetime.timedelta]
    _time_dim_name: ClassVar[str] = "time_utc"

    def __post_init__(self):  # noqa: D105
        # Sanity checks.
        assert self.sample_period_duration, "sample_period_duration cannot be zero length!"
        assert (self.time_periods["start_dt"] < self.time_periods["end_dt"]).all()
        start_dt = pd.DatetimeIndex(self.time_periods["start_dt"])
        assert start_dt.is_monotonic_increasing
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
        """Total number of timesteps per example, including t0.

        Our definition "total_seq_length" is inclusive of both the first and last timestep.
        """
        # Plus 1 because "total_seq_length" is inclusive of both the first and last timestep.
        return int(self.total_duration / self.sample_period_duration) + 1

    @property
    def total_duration(self) -> datetime.timedelta:
        """Total number of duration per example, including t0."""
        return self.history_duration + self.forecast_duration

    @property
    def t0_idx(self) -> int:
        """The index into the array for the most recent observation (t0).

        Remember that, in this code, we consider t0 to be part of the history.
        So, if the history_duration is 1 hour, and sample_period_duration is 30 minutes,
        then "history" will be at indicies 0, 1, and 2.
        """
        return int(self.history_duration / self.sample_period_duration)

    @property
    def datetime_index(self) -> pd.DatetimeIndex:
        """Return a pd.DatetimeIndex of all available datetimes.

        Child classes can override this to, for example, filter out any datetimes
        which don't make sense for this DataSource, e.g. remove nighttime.
        """
        if self.needs_to_load_subset_into_ram:
            data = self.data_on_disk
        else:
            data = self.data_in_ram
        return pd.DatetimeIndex(data[self._time_dim_name])

    def load_subset_into_ram(self, subset_of_contiguous_t0_time_periods: pd.DataFrame) -> None:
        """Load a subset of `data_on_disk` into `data_in_ram`.

        Args:
            subset_of_contiguous_t0_time_periods: DataFrame with columns 'start_dt' and 'end_dt'
                specifying the start and end of value t0 periods.

        Override in DataSources which can only fit a subset of the dataset into RAM.
        """
        _log.info(f"{self.__class__.__name__} load_subset_into_ram().")
        # Delete any existing data in RAM while we're loading new data.
        self._data_in_ram = None

        # Convert t0_time_periods back into the complete time periods we want to load:
        time_periods = self._convert_t0_time_periods_to_periods_to_load(
            subset_of_contiguous_t0_time_periods
        )
        assert (time_periods["start_dt"] < time_periods["end_dt"]).all()

        # Lazily create a new DataArray with just the data we want.
        data_to_load = []
        for _, row in time_periods.iterrows():
            start_dt = row["start_dt"]
            end_dt = row["end_dt"]
            data_for_period = self.data_on_disk.sel({self._time_dim_name: slice(start_dt, end_dt)})
            data_to_load.append(data_for_period)

        # Load into RAM :)
        data_to_load = xr.concat(data_to_load, dim=self._time_dim_name)
        data_to_load = data_to_load.drop_duplicates(dim=self._time_dim_name)
        self._data_in_ram = data_to_load.load()

    def _convert_t0_time_periods_to_periods_to_load(
        self, subset_of_contiguous_t0_time_periods: pd.DataFrame
    ) -> pd.DataFrame:
        """Convert t0_time_periods back into the complete time periods we want to load."""
        time_periods = deepcopy(subset_of_contiguous_t0_time_periods)
        del subset_of_contiguous_t0_time_periods
        time_periods["start_dt"] -= self.history_duration
        time_periods["end_dt"] += self.forecast_duration
        return time_periods

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
        assert (contiguous_time_periods["start_dt"] < contiguous_time_periods["end_dt"]).all()
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
        self, xr_data: xr.DataArray, t0_datetime_utc: datetime.datetime
    ) -> xr.DataArray:
        """Select a timeslice from `xr_data`.

        The returned data does not include an `example` dimension.
        """
        start_dt_ceil = self._get_start_dt_ceil(t0_datetime_utc)
        end_dt_ceil = self._get_end_dt_ceil(t0_datetime_utc)

        # Sanity check!
        assert (
            start_dt_ceil in xr_data.time_utc
        ), f"{start_dt_ceil=} not in xr_data.time_utc! {t0_datetime_utc=}"
        assert (
            end_dt_ceil in xr_data.time_utc
        ), f"{end_dt_ceil=} not in xr_data.time_utc! {t0_datetime_utc=}"

        # Get time slice:
        time_slice = xr_data.sel({self._time_dim_name: slice(start_dt_ceil, end_dt_ceil)})
        self._sanity_check_time_slice(time_slice, self._time_dim_name, t0_datetime_utc)
        return time_slice

    def _sanity_check_time_slice(
        self,
        time_slice: xr.DataArray,
        time_dim_name: str,
        t0_datetime_utc: datetime.datetime,
    ) -> None:
        info_str = f"Context: {t0_datetime_utc=}; {time_dim_name=}; {time_slice=}"
        assert (
            len(time_slice[time_dim_name]) == self.total_seq_length
        ), f"{len(time_slice[time_dim_name])=} != {self.total_seq_length=} {info_str}"
        time_slice_duration = np.timedelta64(
            time_slice[time_dim_name][-1].values - time_slice[time_dim_name][0].values
        )
        expected_duration = np.timedelta64(self.total_duration)
        assert (
            time_slice_duration == expected_duration
        ), f"{time_slice_duration=} != {expected_duration=} {info_str}"

    def _get_start_dt_ceil(
        self, t0_datetime_utc: Union[datetime.datetime, np.datetime64, pd.Timestamp]
    ) -> pd.Timestamp:
        start_dt = pd.Timestamp(t0_datetime_utc) - np.timedelta64(self.history_duration)
        return start_dt.ceil(self.sample_period_duration)

    def _get_end_dt_ceil(
        self, t0_datetime_utc: Union[datetime.datetime, np.datetime64, pd.Timestamp]
    ) -> pd.Timestamp:
        end_dt = pd.Timestamp(t0_datetime_utc) + np.timedelta64(self.forecast_duration)
        return end_dt.ceil(self.sample_period_duration)

    def _set_attributes(self, xr_data: xr.DataArray) -> xr.DataArray:
        xr_data.attrs["t0_idx"] = self.t0_idx
        xr_data.attrs["sample_period_duration"] = self.sample_period_duration
        return xr_data


@dataclass(kw_only=True)
class SpatialDataSource:
    """Abstract base class for dense spatial image data sources (like NWP and satellite).

    Sparse spatial data (such as PV data) doesn't inherit from `SpatialDataSource`.

    Args:
        roi_height_pixels: Height of the image in each example. Must be divisible by 2.
            ROI stands for region of interest.
        roi_width_pixels: Width of the image in each example. Must be divisible by 2.
    """

    roi_height_pixels: int
    roi_width_pixels: int

    # Attributes which are intended to be set for the whole class.
    _y_dim_name: ClassVar[str] = "y"
    _x_dim_name: ClassVar[str] = "x"

    def __post_init__(self):
        assert self.roi_height_pixels > 0, f"{self.roi_height_pixels=} must be > 0!"
        assert self.roi_width_pixels > 0, f"{self.roi_width_pixels=} must be > 0!"
        assert (
            self.roi_height_pixels % 2
        ) == 0, f"{self.roi_height_pixels=} must be divisible by 2!"
        assert (self.roi_width_pixels % 2) == 0, f"{self.roi_width_pixels=} must be divisible by 2!"

    def get_osgb_location_for_example(self) -> Location:
        """Randomly select a valid geographical location for one example.

        Returns:  Location(x_center_osgb, y_center_osgb)
        """
        # Find the minimum and maximum legal values for the randomly sampled x and y positions:
        half_height_of_crop = self.roi_height_pixels // 2
        half_width_of_crop = self.roi_width_pixels // 2

        source_image_height = len(self._data_in_ram[self._y_dim_name])
        source_image_width = len(self._data_in_ram[self._x_dim_name])

        # Plus or minus one for safety.
        min_y = half_height_of_crop + 1
        max_y = source_image_height - half_height_of_crop - 1
        min_x = half_width_of_crop + 1
        max_x = source_image_width - half_width_of_crop - 1

        # Sanity check!
        assert 0 < min_x < source_image_width
        assert 0 < max_x < source_image_width
        assert 0 < min_y < source_image_height
        assert 0 < max_y < source_image_height

        # Randomly pick y and x indexes.
        y_idx = self.rng.integers(low=min_y, high=max_y)
        x_idx = self.rng.integers(low=min_x, high=max_x)

        # Get the OSGB coords for those indexes:
        selected_pixel = self._data_in_ram.isel(
            {
                self._y_dim_name: y_idx,
                self._x_dim_name: x_idx,
            }
        )
        y_osgb = selected_pixel.y_osgb.item()
        x_osgb = selected_pixel.x_osgb.item()

        return Location(x=x_osgb, y=y_osgb)

    def _get_spatial_slice(self, xr_data: xr.DataArray, center_osgb: Location) -> xr.DataArray:
        """Slice `xr_data` to produce a region of interest, centered on `center_osgb`.

        Assume the image data starts top-left.

        The returned data does not include an `example` dimension.
        """
        # Find pixel index at `center_osgb`:
        center_idx = self._get_idx_of_pixel_at_center_of_roi(
            xr_data=xr_data, center_osgb=center_osgb
        )

        # Compute the index for left and right:
        half_height = self.roi_height_pixels // 2
        half_width = self.roi_width_pixels // 2

        left_idx = center_idx.x - half_width
        right_idx = center_idx.x + half_width
        top_idx = center_idx.y - half_height
        bottom_idx = center_idx.y + half_height

        # Sanity check!
        assert left_idx >= 0, f"{left_idx=} must be >= 0!"
        data_width_pixels = len(xr_data[self._x_dim_name])
        assert right_idx <= data_width_pixels, f"{right_idx=} must be <= {data_width_pixels=}"
        assert top_idx >= 0, f"{top_idx=} must be >= 0!"
        data_height_pixels = len(xr_data[self._y_dim_name])
        assert bottom_idx <= data_height_pixels, f"{bottom_idx=} must be <= {data_height_pixels=}"

        selected = xr_data.isel(
            {
                self._x_dim_name: slice(left_idx, right_idx),
                self._y_dim_name: slice(top_idx, bottom_idx),
            }
        )

        # Sanity check:
        assert len(selected[self._x_dim_name]) == self.roi_width_pixels
        assert len(selected[self._y_dim_name]) == self.roi_height_pixels

        return selected

    def _get_idx_of_pixel_at_center_of_roi(
        self, xr_data: xr.DataArray, center_osgb: Location
    ) -> Location:
        """Return x and y index location of pixel at center of region of interest."""
        y_index = xr_data.get_index(self._y_dim_name)
        x_index = xr_data.get_index(self._x_dim_name)
        return Location(
            y=y_index.get_indexer([center_osgb.y], method="nearest")[0],
            x=x_index.get_indexer([center_osgb.x], method="nearest")[0],
        )


@dataclass(kw_only=True)
class ZarrDataSource:
    """Abstract base class for Zarr stores."""

    zarr_path: Union[Path, str]

    def __post_init__(self):
        check_path_exists(self.zarr_path)
