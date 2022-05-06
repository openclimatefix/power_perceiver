import datetime
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import torch
import xarray as xr

from power_perceiver.data_loader.data_loader import BatchKey, NumpyBatch
from power_perceiver.data_loader.satellite import SAT_MEAN, SAT_STD
from power_perceiver.utils import datetime64_to_float

_log = logging.getLogger(__name__)


def get_contiguous_segments(
    dt_index: pd.DatetimeIndex, min_timesteps: int, max_gap: pd.Timedelta = pd.Timedelta("5T")
) -> list[np.ndarray]:
    """Chunk datetime index into contiguous segments, each at least min_timesteps long.

    max_gap defines the threshold for what constitutes a 'gap' between contiguous segments.

    Returns a list of arrays. Each array holds the indicies into `dt_index` of
    one contiguous segment.
    """

    gap_mask = np.diff(dt_index) > max_gap
    gap_indices = np.argwhere(gap_mask)[:, 0]

    # gap_indicies are the indices into dt_index for the timestep immediately before the gap.
    # e.g. if the datetimes at 12:00, 12:05, 18:00, 18:05 then gap_indicies will be [1].
    segment_boundaries = gap_indices + 1

    # Capture the last segment of dt_index.
    segment_boundaries = np.concatenate((segment_boundaries, [len(dt_index)]))

    good_time_idx = []
    start_i = 0
    for end_i in segment_boundaries:
        n_timesteps = end_i - start_i
        if n_timesteps >= min_timesteps:
            good_time_idx.append(np.arange(start_i, end_i))
        start_i = end_i

    return good_time_idx


def _select_data_in_daylight(xr_sat_dataset: xr.Dataset) -> xr.Dataset:
    # TODO: Use angle of the Sun, not hour-of-day.
    time_index = pd.DatetimeIndex(xr_sat_dataset.time)
    daylight_hours_mask = (time_index.hour >= 9) & (time_index.hour <= 15)
    return xr_sat_dataset.isel(time=daylight_hours_mask)


def get_dates(xr_sat_dataset: xr.Dataset) -> np.ndarray:
    return np.sort(np.unique(pd.DatetimeIndex(xr_sat_dataset.time).date))


def num_days(xr_sat_dataset: xr.Dataset) -> int:
    dates = get_dates(xr_sat_dataset)
    return len(dates)


def date_summary_str(xr_sat_dataset: xr.Dataset) -> str:
    time_index = pd.DatetimeIndex(xr_sat_dataset.time)
    return (
        f"there are {num_days(xr_sat_dataset):,d} days of data"
        f" from {time_index[0]} to {time_index[-1]}"
    )


def _select_timesteps_in_contiguous_periods(
    xr_sat_dataset: xr.Dataset, min_timesteps: int
) -> xr.Dataset:
    dt_index = pd.DatetimeIndex(xr_sat_dataset.time)
    good_time_idx = get_contiguous_segments(dt_index, min_timesteps=min_timesteps)
    good_time_idx = np.concatenate(good_time_idx)
    return xr_sat_dataset.isel(time=good_time_idx)


def to_numpy_batch(xr_data: xr.DataArray) -> NumpyBatch:
    example: NumpyBatch = {}
    # Insert a "channels" dimension:
    example[BatchKey.hrvsatellite] = np.expand_dims(xr_data.values, axis=1)
    example[BatchKey.hrvsatellite_time_utc] = datetime64_to_float(xr_data["time"].values)
    for batch_key, dataset_key in (
        (BatchKey.hrvsatellite_y_osgb, "y_osgb"),
        (BatchKey.hrvsatellite_x_osgb, "x_osgb"),
        (BatchKey.hrvsatellite_y_geostationary, "y"),
        (BatchKey.hrvsatellite_x_geostationary, "x"),
    ):
        # HRVSatellite coords are already float32.
        example[batch_key] = xr_data[dataset_key].values
    return example


@dataclass
class SatelliteZarrDataset(torch.utils.data.IterableDataset):
    """Loads data directly from the satellite Zarr store.

    The basic strategy implemented by this class is:

    1. At the start of the epoch, load `n_days_to_load_per_epoch` random days from Zarr into RAM.
    2. During the epoch, randomly sample from those days of satellite data in RAM.
    """

    satellite_zarr_path: Union[Path, str]
    n_days_to_load_per_epoch: int = 128  #: Number of random days to load per epoch.
    n_examples_per_epoch: int = 1024 * 128
    n_timesteps_per_example: int = 31  #: 31 is what's used in v15 of the pre-prepared dataset.
    start_date: datetime.datetime = pd.Timestamp("2020-01-01 00:00")
    end_date: datetime.datetime = pd.Timestamp("2020-12-31 23:59")
    size_pixels: int = 64

    def __post_init__(self):
        super().__init__()
        assert self.end_date > self.start_date

    def per_worker_init(self, worker_id: int = 0) -> None:
        """Called by worker_init_fn on each copy of SatelliteDataset after the
        worker process has been spawned."""
        self.worker_id = worker_id
        # Each worker must have a different seed for its random number generator.
        # Otherwise all the workers will output exactly the same data!
        seed = torch.initial_seed()
        _log.info(f"{worker_id=} has random number generator {seed=:,d}")
        self.rng = np.random.default_rng(seed=seed)

        self._open_satellite_zarr()

    def _open_satellite_zarr(self):
        """Sets `xr_sat_dataset` and `available_dates`."""
        # Get xr.Dataset:
        xr_sat_dataset = xr.open_dataset(
            self.satellite_zarr_path,
            engine="zarr",
            chunks="auto",  # Load the data as a Dask array.
        )
        _log.info("Before any selection, " + date_summary_str(xr_sat_dataset))
        # Select only the timesteps we want:
        xr_sat_dataset = xr_sat_dataset.sel(time=slice(self.start_date, self.end_date))
        xr_sat_dataset = _select_data_in_daylight(xr_sat_dataset)
        xr_sat_dataset = _select_timesteps_in_contiguous_periods(
            xr_sat_dataset, min_timesteps=self.n_timesteps_per_example
        )
        _log.info("After filtering, " + date_summary_str(xr_sat_dataset))
        self.xr_sat_dataset = xr_sat_dataset

    def __iter__(self):
        self._load_random_days_from_disk()  # TODO: Could be done asynchronously
        for _ in range(self.n_examples_per_epoch):
            yield self._get_example()

    def _load_random_days_from_disk(self) -> None:
        """Sets `sat_data_in_mem` and `available_dates`."""
        all_available_dates_on_disk = get_dates(self.xr_sat_dataset)
        days_to_load = self.rng.choice(
            all_available_dates_on_disk, size=self.n_days_to_load_per_epoch, replace=False
        )
        time_index = pd.DatetimeIndex(self.xr_sat_dataset.time)
        mask = np.isin(time_index.date, days_to_load)
        self.sat_data_in_mem = self.xr_sat_dataset.isel(time=mask).load()
        self.available_dates = get_dates(self.sat_data_in_mem)

    def _get_example(self) -> np.ndarray:
        xr_data = self._get_time_slice()
        xr_data = self._get_square(xr_data)
        xr_data = xr_data["data"]
        xr_data = _normalise(xr_data)
        return to_numpy_batch(xr_data)

    def _get_time_slice(self) -> xr.Dataset:
        # Select a random date from the in-memory data
        date = self.rng.choice(self.available_dates)
        sat_data_for_date = self.sat_data_in_mem.sel(
            time=slice(date, date + datetime.timedelta(days=1))
        )

        # Find all the contiguous segments in that day of data and select a random segment:
        contiguous_segments = get_contiguous_segments(
            sat_data_for_date.time, min_timesteps=self.n_timesteps_per_example
        )
        segment_idx = self.rng.integers(low=0, high=len(contiguous_segments))
        segment = contiguous_segments[segment_idx]

        # Pick a random start time within the segment.
        # The +1 is necessary because segment[-1] gives the *index* which might be
        # as low as self.n_timesteps_per_example - 1.
        max_legal_start_idx = segment[-1] - self.n_timesteps_per_example + 1
        assert max_legal_start_idx >= 0, f"{max_legal_start_idx=}"
        if max_legal_start_idx == 0:
            start_idx = 0
        else:
            start_idx = self.rng.integers(low=segment[0], high=max_legal_start_idx)
        end_idx = start_idx + self.n_timesteps_per_example

        return sat_data_for_date.isel(time=slice(start_idx, end_idx))

    def _get_square(self, sat_data_for_date: xr.Dataset) -> xr.Dataset:
        """Pick random square."""
        max_y = len(sat_data_for_date.y) - self.size_pixels
        max_x = len(sat_data_for_date.x) - self.size_pixels
        y_start_idx = self.rng.integers(low=0, high=max_y)
        x_start_idx = self.rng.integers(low=0, high=max_x)
        y_end_idx = y_start_idx + self.size_pixels
        x_end_idx = x_start_idx + self.size_pixels

        return sat_data_for_date.isel(
            y=slice(y_start_idx, y_end_idx), x=slice(x_start_idx, x_end_idx)
        )


def worker_init_fn(worker_id: int):
    """Configures each dataset worker process.

    Just has one job!  To call SatelliteDataset.per_worker_init().
    """
    # get_worker_info() returns information specific to each worker process.
    worker_info = torch.utils.data.get_worker_info()
    dataset_obj = worker_info.dataset  # The Dataset copy in this worker process.
    dataset_obj.per_worker_init(worker_id=worker_id)


def _normalise(sat_data: xr.Dataset) -> xr.Dataset:
    sat_data = sat_data.astype(np.float32)
    sat_data -= SAT_MEAN["HRV"]
    sat_data /= SAT_STD["HRV"]
    return sat_data
