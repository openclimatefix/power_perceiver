import datetime
import logging
from dataclasses import dataclass
from numbers import Number
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.consts import Location
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import (
    NumpyBatch,
    XarrayBatch,
)
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    TimeseriesDataSource,
)
from power_perceiver.time import (
    intersection_of_multiple_dataframes_of_periods,
    time_periods_to_datetime_index,
)
from power_perceiver.utils import (
    sample_row_and_drop_row_from_df,
    set_fsspec_for_multiprocess,
    stack_np_examples_into_batch,
)

_log = logging.getLogger(__name__)


@dataclass
class RawDataset(torch.utils.data.IterableDataset):
    """Dataset for loading data from intermediate data (not pre-prepared batches).

    Initialisation arguments:
        data_source_combos: A dict where the keys are strings (short, arbitrary names
            identifying each combination of DataSources), and the values are tuples or lists of
            instantiated `DataSource` objects. The first data source
            will randomly select the geographical location of each example.

            If a DataSource appears more than once, then you must use different instances for each
            combo (e.g. by using `deepcopy`). Using different instances
            allows each data source combo to load time periods appropriate for that combo.

            For example:
                data_source_kwargs = dict(start_date=start_date, end_date=end_date)
                sat = RawSatelliteDataSource(**data_source_kwargs)
                gsp = RawGSPDataSource(**data_source_kwargs)
                data_source_combos = dict(
                     sat_only=(sat,),
                     gsp_and_sat=(gsp, deepcopy(sat)))

        probability_of_each_combo: A dict where the key is the name of each
            combination of DataSources, and the key is the probability of loading that
            combination of DataSources. Probabilities must sum to 1.
            Optional. If `None` then will use equal probabilities.
        min_duration_to_load_per_epoch: If this an int, then this will be (roughly) the total
            number of hours of contiguous time periods that will be loaded into RAM
            at the start of each epoch. Some DataSources (such as PV) can load the entire
            dataset into RAM at initialization. If using only DataSources which can load
            everything into RAM, then set `min_duration_to_load_per_epoch` to None to avoid
            computing the subset of data to load into RAM at the start of each epoch.
            If `min_duration_to_load_per_epoch` is not `None`, then `ds_combo_for_subsetting` must
            not be `None`.
        load_subset_every_epoch: Set to False for use as a validation dataset.
        n_batches_per_epoch:
        n_examples_per_batch:
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

    data_source_combos: dict[str, Sequence[RawDataSource]]
    probability_of_each_combo: Optional[dict[str, Number]] = None
    min_duration_to_load_per_epoch: Optional[datetime.timedelta] = datetime.timedelta(hours=48 * 12)
    load_subset_every_epoch: bool = True
    n_batches_per_epoch: int = 1024
    n_examples_per_batch: int = 32
    np_batch_processors: Optional[Sequence[Callable]] = None
    t0_freq: str = "5T"

    def __post_init__(self):  # noqa: D105
        self._sanity_check_args()
        super().__init__()
        self._all_t0_periods_per_combo: dict[str, pd.DataFrame] = {}
        self._t0_datetimes_per_combo_for_epoch: dict[str, pd.DatetimeIndex] = {}

    def _sanity_check_args(self):  # noqa: D105
        if self.probability_of_each_combo is not None:
            assert self.data_source_combos.keys() == self.probability_of_each_combo.keys()
        if self.min_duration_to_load_per_epoch is not None:
            assert self.min_duration_to_load_per_epoch > pd.Timedelta(self.t0_freq)
        assert self.n_batches_per_epoch > 0

    def per_worker_init(self, worker_id: int = 0, seed: int = 42) -> None:
        """Called by worker_init_fn on each copy of this dataset after the
        worker process has been spawned."""
        set_fsspec_for_multiprocess()
        self.worker_id = worker_id
        # Each worker must have a different seed for its random number generator.
        # Otherwise all the workers will output exactly the same data!
        # The worker ID will be different for each worker process for each GPU.
        _log.info(f"{worker_id=} has random number generator {seed=:,d}")
        self.rng = np.random.default_rng(seed=seed)

        for data_source in self._unique_data_sources:
            data_source.per_worker_init(worker_id=worker_id, seed=seed)

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

    def __iter__(self) -> Iterator[NumpyBatch]:  # noqa: D105
        self._epoch_start()
        for _ in range(self.n_batches_per_epoch):
            yield self._get_np_batch()

    def _epoch_start(self) -> None:
        if self.load_subset_every_epoch or not self._t0_datetimes_per_combo_for_epoch:
            self._set_t0_datetimes_per_combo_for_epoch_and_maybe_load_subset_into_ram()

    def _set_t0_datetimes_per_combo_for_epoch_and_maybe_load_subset_into_ram(self):
        # Compute subset of contiguous t0 time periods for this epoch, and ask each unique
        # data source that needs to load a subset into RAM to do so:
        data_sources_which_have_loaded = []
        for combo_name, data_sources in self.data_source_combos.items():
            data_sources_which_need_to_load = [
                ds for ds in data_sources if ds.needs_to_load_subset_into_ram
            ]
            if data_sources_which_need_to_load:
                t0_periods_for_combo_for_epoch = self._subset_t0_periods(combo_name)

                # Load into RAM, if necessary:
                for data_source in data_sources_which_need_to_load:
                    # Check we haven't already loaded this data_source in another combo:
                    if not any([data_source is ds for ds in data_sources_which_have_loaded]):
                        data_source.load_subset_into_ram(t0_periods_for_combo_for_epoch)
            else:
                t0_periods_for_combo_for_epoch = self._all_t0_periods_per_combo[combo_name]

            self._t0_datetimes_per_combo_for_epoch[combo_name] = time_periods_to_datetime_index(
                time_periods=t0_periods_for_combo_for_epoch, freq=self.t0_freq
            )

    def _subset_t0_periods(self, combo_name: str) -> pd.DataFrame:
        """Pick a random selection of contiguous time periods for the upcoming epoch.

        The main use-case for this is for `RawSatelliteDataSource` which needs to load a subset of
        data into RAM at the start of each epoch.
        """
        _log.info(f"Selecting a subset of time periods to load into RAM for {combo_name=}.")
        assert (
            self._all_t0_periods_per_combo
        ), "self._all_t0_periods_per_combo is empty! Have you forgotten to call per_worker_init?"
        all_t0_periods_for_combo = self._all_t0_periods_per_combo[combo_name]

        # Select random periods. We use a `while` loop instead of just doing
        # `rng.choice(all_t0_periods_for_combo, size=n)` because using `rng.choice`
        # wouldn't take into consideration that some periods are longer than others,
        # and we want to load roughly the same duration of data per epoch.
        random_t0_periods: list[pd.Series[str, pd.Timestamp]] = []
        total_duration_of_periods = pd.Timedelta(0)
        while total_duration_of_periods < self.min_duration_to_load_per_epoch:
            period, all_t0_periods_for_combo = sample_row_and_drop_row_from_df(
                all_t0_periods_for_combo, rng=self.rng
            )
            random_t0_periods.append(period)
            period_duration = period.end_dt - period.start_dt
            total_duration_of_periods += period_duration
            if all_t0_periods_for_combo.empty:
                break

        random_t0_periods = pd.DataFrame(random_t0_periods).sort_values("start_dt")
        _log.info(
            f"{self.worker_id=}. Selected {len(random_t0_periods):,d} random periods,"
            f" with total duration = {total_duration_of_periods},"
            f"from {random_t0_periods.iloc[0].start_dt} to {random_t0_periods.iloc[-1].end_dt}"
        )
        return random_t0_periods

    def _get_np_batch(self) -> NumpyBatch:
        np_examples = [self._get_np_example() for _ in range(self.n_examples_per_batch)]
        np_batch = stack_np_examples_into_batch(np_examples)
        return self._process_np_batch(np_batch)

    def _get_np_example(self) -> NumpyBatch:
        xr_example = self._get_xr_example()
        return self._xarray_to_numpy_example(xr_example)

    def _get_xr_example(self) -> XarrayBatch:
        chosen_combo_name = self._choose_combo_name()
        t0_datetime_utc = self._choose_t0_datetime(chosen_combo_name)
        location_osgb = self._choose_osgb_location(chosen_combo_name)

        try:
            xr_example = self._get_specific_xr_example(
                chosen_combo_name=chosen_combo_name,
                t0_datetime_utc=t0_datetime_utc,
                location_osgb=location_osgb,
            )
        except Exception as e:
            raise e.__class__(f"{chosen_combo_name=}; {t0_datetime_utc=}; {location_osgb=}") from e

        return xr_example

    def _choose_combo_name(self) -> str:
        """Pick a random ds_combo_name using the probabilities."""
        data_source_combo_names = list(self.data_source_combos.keys())
        if self.probability_of_each_combo is None:
            prob_of_each_combo = None
        else:
            # Need to ensure the probabilities are in the same order as the combos!
            prob_of_each_combo = [
                self.probability_of_each_combo[combo_name] for combo_name in data_source_combo_names
            ]
        return self.rng.choice(data_source_combo_names, p=prob_of_each_combo)

    def _choose_t0_datetime(self, chosen_combo_name: str) -> datetime.datetime:
        return self.rng.choice(self._t0_datetimes_per_combo_for_epoch[chosen_combo_name])

    def _choose_osgb_location(self, chosen_combo_name: str) -> Location:
        data_source_combo = self.data_source_combos[chosen_combo_name]
        data_source_which_selects_location = data_source_combo[0]
        return data_source_which_selects_location.get_osgb_location_for_example()

    def _get_specific_xr_example(
        self,
        chosen_combo_name: str,
        t0_datetime_utc: datetime.datetime,
        location_osgb: Location,
    ) -> XarrayBatch:
        # Loop through each data source in the combo:
        chosen_data_source_combo = self.data_source_combos[chosen_combo_name]
        xr_example: XarrayBatch = {}
        for data_source in self._unique_data_sources:
            if data_source in chosen_data_source_combo:
                xr_example[data_source.__class__] = data_source.get_example(
                    t0_datetime_utc, location_osgb
                )
            else:
                xr_example[data_source.__class__] = data_source.empty_example

        return xr_example

    def _xarray_to_numpy_example(self, xr_example: XarrayBatch) -> NumpyBatch:
        """Convert from xarray Datasets to numpy."""
        np_example: NumpyBatch = {}
        for data_loader_class, xr_dataset in xr_example.items():
            if data_loader_class == BatchKey.requested_timesteps:
                # `ReduceNumTimesteps` introduces a `requested_timesteps` key,
                # whose value is a np.ndarray.
                requested_timesteps = xr_dataset
                np_example[BatchKey.requested_timesteps] = requested_timesteps
            else:
                np_data_for_data_source = data_loader_class.to_numpy(xr_dataset)
                np_example.update(np_data_for_data_source)
        return np_example

    def _process_np_batch(self, np_batch: NumpyBatch) -> NumpyBatch:
        """If necessary, do any processing which needs to be done across modalities,
        on the NumpyBatch."""
        if self.np_batch_processors:
            for np_batch_processor in self.np_batch_processors:
                np_batch = np_batch_processor(np_batch)
        return np_batch

    @property
    def _unique_data_sources(self) -> list[RawDataSource]:
        # We can't use `set()` or `np.unique()` on a list of `RawDataSource` objects.
        unique_data_sources: list[RawDataSource] = []
        for tuple_of_data_sources in self.data_source_combos.values():
            for data_source in tuple_of_data_sources:
                # We need to use `is` to check if data_source *is* the exact same instance
                # as a RawDataSource already in `unique_data_sources`.
                # Using `if data_source in unique_data_sources` doesn't work because
                # that just uses the equality operator, which returns True if two
                # objects are identical but not necessarily the same instance.
                if not any([data_source is ds for ds in unique_data_sources]):
                    unique_data_sources.append(data_source)
        return unique_data_sources
