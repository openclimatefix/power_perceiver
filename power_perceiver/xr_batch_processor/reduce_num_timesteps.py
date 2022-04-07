from dataclasses import dataclass
from typing import Iterable, Union

import numpy as np

from power_perceiver.data_loader.data_loader import XarrayBatch


@dataclass
class ReduceNumTimesteps:
    """Reduce the number of timesteps per example to `requested_timesteps`.

    If `requested_timesteps` is an int then randomly pick different timesteps for each batch.

    If `requested_timesteps` is an array of ints then use that.
    """

    requested_timesteps: Union[int, Iterable[int]]
    num_timesteps_available: int = 31

    def __post_init__(self):
        self.rng = np.random.default_rng()

    def __call__(self, xr_batch: XarrayBatch) -> XarrayBatch:
        if isinstance(self.requested_timesteps, int):
            requested_timesteps = self.rng.integers(
                low=0, high=self.num_timesteps_available, size=self.requested_timesteps
            )
            print(requested_timesteps)
        else:
            requested_timesteps = self.requested_timesteps

        for name, xr_dataset in xr_batch.items():
            xr_batch[name] = xr_dataset.isel(time=requested_timesteps)
        return xr_batch
