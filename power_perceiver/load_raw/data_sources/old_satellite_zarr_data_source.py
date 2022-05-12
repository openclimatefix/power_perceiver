import datetime
import itertools
import logging
from dataclasses import dataclass
from numbers import Number
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
import pandas as pd
import pvlib
import torch
import xarray as xr

from power_perceiver.geospatial import osgb_to_lat_lon
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import (
    BatchKey,
    NumpyBatch,
)
from power_perceiver.load_prepared_batches.data_sources.satellite import SAT_MEAN, SAT_STD
from power_perceiver.utils import datetime64_to_float

_log = logging.getLogger(__name__)


@dataclass
class OLDSatelliteZarrDataset(torch.utils.data.IterableDataset):

    n_examples_per_epoch: int = 1024 * 32

    def __post_init__(self):
        super().__init__()

    def __iter__(self):
        if self.data_in_ram is None or not self.load_once:
            self._load_random_days_from_disk()  # TODO: Could be done asynchronously
        for _ in range(self.n_examples_per_epoch):
            yield self._get_example()

    def _load_random_days_from_disk(self) -> None:
        """Sets `sat_data_in_mem` and `available_dates`."""
        self.data_in_ram = None  # Remove previous data from memory.
        all_available_dates_on_disk = get_dates(self.xr_sat_dataset)
        days_to_load = self.rng.choice(
            all_available_dates_on_disk, size=self.n_days_to_load_per_epoch, replace=False
        )
        time_index = pd.DatetimeIndex(self.xr_sat_dataset.time)
        mask = np.isin(time_index.date, days_to_load)
        self.data_in_ram = self.xr_sat_dataset.isel(time=mask).load()
        self.available_dates = get_dates(self.data_in_ram)

    def _get_example(self) -> NumpyBatch:
        xr_data = self._get_time_slice()
        xr_data = self._get_square(xr_data)
        xr_data = xr_data["data"]
        xr_data = xr_data.expand_dims(dim="example", axis=0)
        xr_data = _normalise(xr_data)
        np_batch = to_numpy_batch(xr_data)
        if self.np_batch_processors:
            for batch_processor in self.np_batch_processors:
                np_batch = batch_processor(np_batch)

        # Remove example dim:
        for key, array in np_batch.items():
            np_batch[key] = array[0]
        return np_batch

    def _get_time_slice(self) -> xr.Dataset:
        # Select a random date from the in-memory data
        date = self.rng.choice(self.available_dates)
        sat_data_for_date = self.data_in_ram.sel(
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
