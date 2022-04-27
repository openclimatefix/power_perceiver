from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from power_perceiver.data_loader.data_loader import DataLoader, XarrayBatch


@dataclass
class ReduceNumTimesteps:
    """Reduce the number of timesteps per example to `requested_timesteps`.

    If `requested_timesteps` is an int then randomly pick `requested_timesteps` different
    timesteps for each batch.

    If `requested_timesteps` is an array of ints then use that array as the index into
    each xr_dataset.
    """

    num_requested_history_timesteps: int = 2
    num_requested_forecast_timesteps: int = 4

    num_history_timesteps_available: int = 12
    num_total_timesteps_available: int = 31

    keys: Optional[Iterable[DataLoader]] = None

    def __post_init__(self):
        # Any xr_batch_processor with an `rng` attribute will have the
        # rng seeded by the `seed_rngs` `worker_init_function`.
        self.rng = np.random.default_rng()

    def __call__(self, xr_batch: XarrayBatch) -> XarrayBatch:
        num_total_requested_timesteps = (
            self.num_requested_history_timesteps + self.num_requested_forecast_timesteps
        )
        requested_timesteps = np.empty(num_total_requested_timesteps, dtype=np.int32)
        requested_timesteps[
            : self.num_requested_history_timesteps
        ] = self._random_int_without_replacement(
            start=0,
            stop=self.num_history_timesteps_available,
            num=self.num_requested_history_timesteps,
        )

        requested_timesteps[
            self.num_requested_history_timesteps :
        ] = self._random_int_without_replacement(
            start=self.num_history_timesteps_available,
            stop=self.num_total_timesteps_available,
            num=self.num_requested_forecast_timesteps,
        )

        if self.keys is None:
            keys = xr_batch.keys()
        else:
            keys = self.keys

        for key in keys:
            xr_dataset = xr_batch[key]
            xr_batch[key] = xr_dataset.isel(time=requested_timesteps)
        return xr_batch

    def _random_int_without_replacement(self, start: int, stop: int, num: int) -> np.ndarray:
        # This seems to be the best way to get random ints *without* replacement:
        ints = self.rng.choice(np.arange(start=start, stop=stop), size=num, replace=False)
        return np.sort(ints)
