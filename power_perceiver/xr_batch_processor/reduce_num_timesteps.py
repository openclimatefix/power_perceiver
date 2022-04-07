from dataclasses import dataclass
from typing import Iterable

from power_perceiver.data_loader.data_loader import XarrayBatch


@dataclass
class ReduceNumTimesteps:
    """Reduce the number of timesteps per example to `requested_timesteps`."""

    requested_timesteps: Iterable[int]

    def __call__(self, xr_batch: XarrayBatch) -> XarrayBatch:
        for name, xr_dataset in xr_batch.items():
            xr_batch[name] = xr_dataset.isel(time=self.requested_timesteps)
        return xr_batch
