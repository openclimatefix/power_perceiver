import logging
from dataclasses import dataclass
from numbers import Number
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
import torch

from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    TimeseriesDataSource,
)
from power_perceiver.time import (
    intersection_of_2_dataframes_of_periods,
    intersection_of_multiple_dataframes_of_periods,
)

_log = logging.getLogger(__name__)


@dataclass
class RawDataset(torch.utils.data.Dataset):
    """Dataset for loading data from intermediate data (not pre-prepared batches).

    Initialisation arguments:
        data_source_combos: A dict where the keys are strings (short, arbitrary names
            identifying each combination of DataSources), and the values are tuples or lists of
            instantiated `DataSource` objects. The first data source
            will randomly select the geographical location of each example. Note that,
            if a DataSource appears more than once, then please use the same instance
            multiple times: For example:
                data_source_kwargs = dict(start_date=start_date, end_date=end_date)
                sat = RawSatelliteDataSource(**data_source_kwargs)
                pv = RawPVDataSource(**data_source_kwargs)
                data_source_combos = dict(sat_only=(sat,), sat_and_pv=(sat, pv))
        probability_of_each_combo: A dict where the key is the name of each
            combination of DataSources, and the key is the probability of loading that
            combination of DataSources. Probabilities must sum to 1.
        ds_combo_for_subsetting: The name of the DataSource combination that will provide the
            set of time periods that will be randomly sampled from at the start of each epoch
            when deciding which time periods to load into RAM. For example, let's say we have
            two DataSource combinations: 'sat_only' and 'sat_and_pv'. Let's say we have more
            years of satellite data than PV data. When creating 'sat_only' examples, we
            want to sample from *all* the satellite data. We don't want to limit ourselves to only
            sampling from the intersection between the time periods available for satellite and PV.
            But the satellite data is too large to load into RAM at initialization. So, at the
            start of each epoch, we load a random subset of satellite data into RAM. So we'd set
            `ds_combo_for_subsetting='sat_only'`. Set `ds_combo_for_subsetting` to None if
            no DataSources need to load a subset of data into RAM at the start of each epoch.
            If `ds_combo_for_subsetting` is not None, then `n_hours_to_load_per_epoch` must
            not be `None`.
        n_hours_to_load_per_epoch: If this an int, then this will be (roughly) the total
            number of hours of contiguous time periods that will be loaded into RAM
            at the start of each epoch. Some DataSources (such as PV) can load the entire
            dataset into RAM at initialization. If using only DataSources which can load
            everything into RAM, then set `n_hours_to_load_per_epoch` to None to avoid
            computing the subset of data to load into RAM at the start of each epoch.
            If `n_hours_to_load_per_epoch` is not `None`, then `ds_combo_for_subsetting` must
            not be `None`.
        load_subset_every_epoch: Set to False for use as a validation dataset.
        n_examples_per_epoch:
        xr_batch_processors: Functions which takes an XarrayBatch, and processes
            *across* modalities, and returns the processed XarrayBatch. Note that xarray
            processing *within* a modality should be done in `DataSource.transforms`.
        np_batch_processors: Functions which takes a NumpyBatch, and processes
            *across* modalities, and returns the processed NumpyBatch. Note that numpy
            processing *within* a modality should be done in `DataSource.to_numpy`.
        t0_freq: The temporal frequency at which to convert time periods to datetimes.

    Attributes:
        _all_t0_periods_per_combo: dict[str, pd.DataFrame] where each row of each DataFrame
            represents a single time period. Each pd.DataFrame has two columns:
            `start_dt` and `end_dt` (where 'dt' is short for 'datetime').
            The keys of the dict are the same short names of the combination of DataSources.
            This is the total t0 time periods available, before any subsetting to load into RAM.
        _t0_datetimes_per_combo_for_epoch: dict[str, pd.DatetimeIndex]
    """

    data_source_combos: dict[str, Iterable[RawDataSource]]
    probability_of_each_combo: dict[str, Number]
    ds_combo_for_subsetting: Optional[str] = None
    n_hours_to_load_per_epoch: Optional[int] = 48 * 12
    load_subset_every_epoch: bool = True
    n_examples_per_epoch: int = 1024 * 32
    xr_batch_processors: Optional[Iterable[Callable]] = None
    np_batch_processors: Optional[Iterable[Callable]] = None
    t0_freq: str = "5T"

    def __post_init__(self):  # noqa: D105
        self._sanity_check_args()
        super().__init__()
        self._all_t0_periods_per_combo: dict[str, pd.DataFrame] = {}
        self._t0_datetimes_per_combo_for_epoch: dict[str, pd.DatetimeIndex] = {}

    def _sanity_check_args(self):  # noqa: D105
        if self.ds_combo_for_subsetting is not None:
            assert self.ds_combo_for_subsetting in self.data_source_combos
            assert self.n_hours_to_load_per_epoch is not None
        assert self.data_source_combos.keys() == self.probability_of_each_combo.keys()
        if self.n_hours_to_load_per_epoch is not None:
            assert self.n_hours_to_load_per_epoch > 0
            assert self.ds_combo_for_subsetting is not None
        assert self.n_examples_per_epoch > 0

    def per_worker_init(self, worker_id: int = 0) -> None:
        """Called by worker_init_fn on each copy of this dataset after the
        worker process has been spawned."""
        self.worker_id = worker_id
        # Each worker must have a different seed for its random number generator.
        # Otherwise all the workers will output exactly the same data!
        seed = torch.initial_seed()
        _log.info(f"{worker_id=} has random number generator {seed=:,d}")
        self.rng = np.random.default_rng(seed=seed)

        for data_source in self._unique_data_sources:
            data_source.per_worker_init(worker_id=worker_id)

        self._get_intersection_of_all_t0_periods_per_combo()

    def _get_intersection_of_all_t0_periods_per_combo(self) -> None:
        """Get intersection of all contig t0 periods per combo, before subsetting.

        Sets `self._all_t0_periods_per_combo`
        """
        self._all_t0_periods_per_combo = {}
        for combo_name, data_sources in self.data_source_combos.items():
            t0_periods_for_combo: list[pd.DataFrame] = []
            for data_source in data_sources:
                if isinstance(data_source, TimeseriesDataSource):
                    t0_periods_for_combo.append(data_source.get_contiguous_t0_time_periods())
            self._all_t0_periods_per_combo[
                combo_name
            ] = intersection_of_multiple_dataframes_of_periods(t0_periods_for_combo)

    def __iter__(self):  # noqa: D105
        if self.load_subset_every_epoch or not self._t0_datetimes_per_combo_for_epoch:
            self._set_t0_datetimes_per_combo_for_epoch_and_maybe_load_subset_into_ram()
        for _ in range(self.n_examples_per_epoch):
            yield self._get_example()

    def _set_t0_datetimes_per_combo_for_epoch_and_maybe_load_subset_into_ram(self):
        if self.ds_combo_for_subsetting is None:
            _log.info("`ds_combo_for_subsetting` is None, so don't need to load subset into RAM.")
            if not self._t0_datetimes_per_combo_for_epoch:
                self._t0_datetimes_per_combo_for_epoch = time_periods_to_datetimes_per_combo(
                    self._all_t0_periods_per_combo, freq=self.t0_freq
                )
            return

        # Compute subset of contiguous t0 time periods for this epoch, and ask each unique data source
        # that needs to load a subset into RAM to do so:
        subset_of_t0_periods_for_epoch = self._subset_t0_periods()
        for data_source in self._unique_data_sources:
            if data_source.needs_to_load_subset_into_ram:
                # Ensure we only ask this data_source to load into RAM data that it has available:
                subset_for_ds = intersection_of_2_dataframes_of_periods(
                    subset_of_t0_periods_for_epoch, data_source.get_contiguous_t0_time_periods()
                )
                data_source.load_subset_into_ram(subset_for_ds)

        # For each data source combo, we need to find the intersection of time periods
        # between `subset_of_t0_periods_for_epoch` and the total time periods available for that combo.
        subset_of_t0_periods_per_combo_for_epoch: dict[str, pd.DataFrame] = {}
        for combo_name, all_t0_periods_for_combo in self._all_t0_periods_per_combo.items():
            subset_of_t0_periods_per_combo_for_epoch[
                combo_name
            ] = intersection_of_2_dataframes_of_periods(
                all_t0_periods_for_combo, subset_of_t0_periods_for_epoch
            )
        self._t0_datetimes_per_combo_for_epoch = time_periods_to_datetimes_per_combo(
            subset_of_t0_periods_per_combo_for_epoch, freq=self.t0_freq
        )

    def _subset_t0_periods(self) -> pd.DataFrame:
        """Pick a random selection of contiguous time periods for the upcoming epoch.

        The main use-case for this is for SatelliteDataSource which needs to load a subset of data
        into RAM at the start of each epoch.
        """
        _log.info(
            f"Using {self.ds_combo_for_subsetting=} to select a subset of time periods to load"
            " into RAM."
        )
        assert self._all_t0_periods_per_combo
        all_t0_periods_for_combo = self._all_t0_periods_per_combo[self.ds_combo_for_subsetting]
        # TODO:
        # While loop:
        #   select random time period from all_t0_periods_for_combo
        #   append that to the list of random time periods
        #   if the total length of selected time periods > self.n_hours_to_load_per_epoch then break
        # Turn the list of periods into a DataFrame and return!

    def _get_example(self) -> NumpyBatch:
        # TODO!
        # Pick a random ds_combo_name using the probabilities.
        # Randomly sample t0_dt from self._t0_datetimes_per_combo_for_epoch[data_source_combo]
        # data_source_combo = self.data_source_combos[ds_combo_name]
        # data_source_which_selects_location = data_source_combo[0]
        # location = data_source_which_selects_location.get_location_osgb_for_example()
        # xr_batch: XarrayBatch = {}
        # for data_source in data_source_combo:
        #     example_from_ds = get_example(t0_datetime, location)
        #     xr_batch[data_source.__class__] = example_from_ds
        # Loop round the other data sources calling get_empty_example().
        # xr_batch_processors
        # to_numpy()
        # np_batch_processors
        # Return!
        pass

    @property
    def _unique_data_sources(self):
        data_sources = []
        for tuple_of_data_sources in self.data_source_combos.values():
            for data_source in tuple_of_data_sources:
                data_sources.append(data_source)
        return np.unique(data_sources)
