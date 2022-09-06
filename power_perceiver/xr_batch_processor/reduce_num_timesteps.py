from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import (
    PreparedDataSource,
    XarrayBatch,
)


@dataclass
class ReduceNumTimesteps:
    """Reduce the number of timesteps per example to `requested_timesteps`.

    NOT ACTUALLY USED BY THE NEW TRAINING REGIME. INSTEAD WE SUBSELECT TIMESTEPS
    WITHIN THE MODEL ITSELF. THIS FOR TWO REASONS: 1) IT'S TRICKY TO USE
    XR_BATCH_PROCESSORS IN THE RawDataset. 2) WE WANT THE SatellitePredictor
    TO ALWAYS USE THE FULL SET OF TIMESTEPS.

    If `requested_timesteps` is an int then randomly pick `requested_timesteps` different
    timesteps for each batch.

    If `requested_timesteps` is an array of ints then use that array as the index into
    each xr_dataset.
    """

    num_requested_history_timesteps: int = 7  # Has to be total number available when using UNet.
    num_requested_forecast_timesteps: int = 8

    num_history_timesteps_available: int = 7
    num_total_timesteps_available: int = 31

    keys: Optional[Iterable[PreparedDataSource]] = None

    def __post_init__(self):
        # Any xr_batch_processor with an `rng` attribute will have the
        # rng seeded by the `seed_rngs` `worker_init_function`.
        self.rng = np.random.default_rng()

    def __call__(self, xr_batch: XarrayBatch) -> XarrayBatch:
        num_total_requested_timesteps = (
            self.num_requested_history_timesteps + self.num_requested_forecast_timesteps
        )
        # Need to use int64 so PyTorch can use requested_timesteps as an index.
        requested_timesteps = np.empty(num_total_requested_timesteps, dtype=np.int64)
        requested_timesteps[
            : self.num_requested_history_timesteps
        ] = random_int_without_replacement(
            start=0,
            stop=self.num_history_timesteps_available,
            num=self.num_requested_history_timesteps,
            rng=self.rng,
        )

        requested_timesteps[
            self.num_requested_history_timesteps :
        ] = random_int_without_replacement(
            start=self.num_history_timesteps_available,
            stop=self.num_total_timesteps_available,
            num=self.num_requested_forecast_timesteps,
            rng=self.rng,
        )

        if self.keys is None:
            keys = xr_batch.keys()
        else:
            keys = self.keys

        for key in keys:
            xr_dataset = xr_batch[key]
            xr_batch[key] = xr_dataset.isel(time=requested_timesteps)

        xr_batch[BatchKey.requested_timesteps] = requested_timesteps

        return xr_batch


def random_int_without_replacement(
    start: int, stop: int, num: int, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """This seems to be the best way to get random ints *without* replacement"""
    if rng is None:
        rng = np.random.default_rng()
    ints = rng.choice(np.arange(start=start, stop=stop), size=num, replace=False)
    return np.sort(ints)
