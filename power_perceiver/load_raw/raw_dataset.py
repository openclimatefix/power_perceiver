import datetime
import logging
from dataclasses import dataclass
from numbers import Number
from typing import Callable, Iterator, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from power_perceiver.consts import BatchKey, Location
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import (
    NumpyBatch,
    XarrayBatch,
)
from power_perceiver.load_raw.data_sources.raw_data_source import (
    RawDataSource,
    TimeseriesDataSource,
)
from power_perceiver.time import (
    intersection_of_2_dataframes_of_periods,
    intersection_of_multiple_dataframes_of_periods,
    time_periods_to_datetime_index,
)
from power_perceiver.utils import sample_row_and_drop_row_from_df

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
            Optional. If `None` then will use equal probabilities.
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
            If `ds_combo_for_subsetting` is not None, then `min_duration_to_load_per_epoch` must
            not be `None`.
        min_duration_to_load_per_epoch: If this an int, then this will be (roughly) the total
            number of hours of contiguous time periods that will be loaded into RAM
            at the start of each epoch. Some DataSources (such as PV) can load the entire
            dataset into RAM at initialization. If using only DataSources which can load
            everything into RAM, then set `min_duration_to_load_per_epoch` to None to avoid
            computing the subset of data to load into RAM at the start of each epoch.
            If `min_duration_to_load_per_epoch` is not `None`, then `ds_combo_for_subsetting` must
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

    data_source_combos: dict[str, Sequence[RawDataSource]]
    probability_of_each_combo: Optional[dict[str, Number]] = None
    ds_combo_for_subsetting: Optional[str] = None
    min_duration_to_load_per_epoch: Optional[datetime.timedelta] = datetime.timedelta(hours=48 * 12)
    load_subset_every_epoch: bool = True
    n_examples_per_epoch: int = 1024 * 32
    xr_batch_processors: Optional[Sequence[Callable]] = None
    np_batch_processors: Optional[Sequence[Callable]] = None
    t0_freq: str = "5T"

    def __post_init__(self):  # noqa: D105
        self._sanity_check_args()
        super().__init__()
        self._all_t0_periods_per_combo: dict[str, pd.DataFrame] = {}
        self._t0_datetimes_per_combo_for_epoch: dict[str, pd.DatetimeIndex] = {}

    def _sanity_check_args(self):  # noqa: D105
        if self.ds_combo_for_subsetting is not None:
            assert self.ds_combo_for_subsetting in self.data_source_combos
            assert self.min_duration_to_load_per_epoch is not None
        if self.probability_of_each_combo is not None:
            assert self.data_source_combos.keys() == self.probability_of_each_combo.keys()
        if self.min_duration_to_load_per_epoch is not None:
            assert self.min_duration_to_load_per_epoch > pd.Timedelta(self.t0_freq)
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

    def __iter__(self) -> Iterator[NumpyBatch]:  # noqa: D105
        if self.load_subset_every_epoch or not self._t0_datetimes_per_combo_for_epoch:
            self._set_t0_datetimes_per_combo_for_epoch_and_maybe_load_subset_into_ram()
        for _ in range(self.n_examples_per_epoch):
            yield self._get_example()

    def _set_t0_datetimes_per_combo_for_epoch_and_maybe_load_subset_into_ram(self):
        if self.ds_combo_for_subsetting is None:
            _log.info("`ds_combo_for_subsetting` is None, so don't need to load subset into RAM.")
            if not self._t0_datetimes_per_combo_for_epoch:
                self._t0_datetimes_per_combo_for_epoch = _time_periods_to_datetimes_per_combo(
                    self._all_t0_periods_per_combo, freq=self.t0_freq
                )
            return

        # Compute subset of contiguous t0 time periods for this epoch, and ask each unique
        # data source that needs to load a subset into RAM to do so:
        subset_of_t0_periods_for_epoch = self._subset_t0_periods()
        for data_source in self._unique_data_sources:
            if data_source.needs_to_load_subset_into_ram:
                # Ensure we only ask this data_source to load into RAM data that it has available:
                subset_for_ds = intersection_of_2_dataframes_of_periods(
                    subset_of_t0_periods_for_epoch, data_source.get_contiguous_t0_time_periods()
                )
                data_source.load_subset_into_ram(subset_for_ds)

        # For each data source combo, we need to find the intersection of time periods
        # between `subset_of_t0_periods_for_epoch` and the total time periods available
        # for that combo.
        subset_of_t0_periods_per_combo_for_epoch: dict[str, pd.DataFrame] = {}
        for combo_name, all_t0_periods_for_combo in self._all_t0_periods_per_combo.items():
            subset_for_combo = intersection_of_2_dataframes_of_periods(
                all_t0_periods_for_combo, subset_of_t0_periods_for_epoch
            )
            subset_of_t0_periods_per_combo_for_epoch[combo_name] = subset_for_combo
        self._t0_datetimes_per_combo_for_epoch = _time_periods_to_datetimes_per_combo(
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
        assert self.ds_combo_for_subsetting
        all_t0_periods_for_combo = self._all_t0_periods_per_combo[self.ds_combo_for_subsetting]

        # Select random periods. We use a `while` loop instead of just doing
        # `rng.choice(all_t0_periods_for_combo, size=n)` because using `rng.choice`
        # wouldn't take into consideration that some periods are longer than others,
        # and we want to load roughly the same duration of data per epoch.
        random_t0_periods: list[pd.Series[str, pd.Timestamp]] = []
        total_duration_of_periods: pd.Timedelta = pd.Timedelta(0)
        while total_duration_of_periods < self.min_duration_to_load_per_epoch:
            period, all_t0_periods_for_combo = sample_row_and_drop_row_from_df(
                all_t0_periods_for_combo, rng=self.rng
            )
            random_t0_periods.append(period)
            period_duration = period.end_dt - period.start_dt
            total_duration_of_periods += period_duration

        _log.info(
            f"Selected {len(random_t0_periods):,d} random periods,"
            f" with total duration = {total_duration_of_periods}"
        )
        return pd.DataFrame(random_t0_periods).sort_values("start_dt")

    def _get_example(self) -> NumpyBatch:
        chosen_combo_name = self._randomly_choose_combo_name()

        # Randomly sample t0 and location:
        t0_datetime_utc = self.rng.choice(self._t0_datetimes_per_combo_for_epoch[chosen_combo_name])
        location_osgb = self._randomly_choose_osgb_location(chosen_combo_name)

        xr_example = self._get_xarray_example(
            chosen_combo_name=chosen_combo_name,
            t0_datetime_utc=t0_datetime_utc,
            location_osgb=location_osgb,
        )

        # TODO: Tell the ML model which type of "combo" this is.
        xr_example = self._process_xr_example(xr_example)
        np_example = self._xarray_to_numpy_example(xr_example)
        del xr_example
        np_example = self._process_np_example(np_example)
        return np_example

    def _randomly_choose_combo_name(self) -> str:
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

    def _randomly_choose_osgb_location(self, chosen_combo_name: str) -> Location:
        data_source_combo = self.data_source_combos[chosen_combo_name]
        data_source_which_selects_location = data_source_combo[0]
        return data_source_which_selects_location.get_osgb_location_for_example()

    def _get_xarray_example(
        self,
        chosen_combo_name: str,
        t0_datetime_utc: datetime.datetime,
        location_osgb: Location,
    ) -> XarrayBatch:
        # Loop through each data source in the combo:
        data_source_combo = self.data_source_combos[chosen_combo_name]
        xr_batch: XarrayBatch = {}
        for data_source in data_source_combo:
            example_from_ds = data_source.get_example(t0_datetime_utc, location_osgb)
            xr_batch[data_source.__class__] = example_from_ds

        # TODO: Loop round the other data sources calling get_empty_example().
        return xr_batch

    def _process_xr_example(self, xr_example: XarrayBatch) -> XarrayBatch:
        """If necessary, do any processing which needs to be done across modalities,
        on the xr.Datasets."""
        if self.xr_batch_processors:
            for xr_batch_processor in self.xr_batch_processors:
                xr_example = xr_batch_processor(xr_example)
        return xr_example

    def _xarray_to_numpy_example(self, xr_example: XarrayBatch) -> NumpyBatch:
        """Convert from xarray Datasets to numpy."""
        np_batch: NumpyBatch = {}
        for data_loader_class, xr_dataset in xr_example.items():
            if data_loader_class == BatchKey.requested_timesteps:
                # `ReduceNumTimesteps` introduces a `requested_timesteps` key,
                # whose value is a np.ndarray.
                requested_timesteps = xr_dataset
                np_batch[BatchKey.requested_timesteps] = requested_timesteps
            else:
                np_data_for_data_source = data_loader_class.to_numpy(xr_dataset)
                np_batch.update(np_data_for_data_source)
        return np_batch

    def _process_np_example(self, np_example: NumpyBatch) -> NumpyBatch:
        """If necessary, do any processing which needs to be done across modalities,
        on the NumpyBatch."""
        if self.np_batch_processors:
            for np_batch_processor in self.np_batch_processors:
                np_example = np_batch_processor(np_example)
        return np_example

    @property
    def _unique_data_sources(self):
        data_sources = []
        for tuple_of_data_sources in self.data_source_combos.values():
            for data_source in tuple_of_data_sources:
                data_sources.append(data_source)
        return np.unique(data_sources)


def _time_periods_to_datetimes_per_combo(
    time_periods_per_combo: dict[str, pd.DataFrame], freq=str
) -> dict[str, pd.DatetimeIndex]:
    """Convert a dict of time periods to a dict of pd.DatetimeIndexes.

    See the docstring for `power_perceiver.time.tim_periods_to_datetime_index` for more info.

    Args:
        time_periods_per_combo: dict where keys are the data source combination name
            and values are a DataFrame with columns ['start_dt', 'end_dt'].
        freq: str

    Returns: dict where keys are the same as the keys for `time_periods_per_combo`,
        and each value is a `pd.DatetimeIndex`.
    """
    datetimes_per_combo: dict[str, pd.DatetimeIndex] = {}
    for combo_name, time_periods in time_periods_per_combo.items():
        datetimes_per_combo[combo_name] = time_periods_to_datetime_index(
            time_periods=time_periods, freq=freq
        )
    return datetimes_per_combo
