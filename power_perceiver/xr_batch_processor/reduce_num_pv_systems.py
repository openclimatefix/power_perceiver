from dataclasses import dataclass

import numpy as np
import xarray as xr

from power_perceiver.load_prepared_batches.data_sources import PV
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import XarrayBatch


@dataclass
class ReduceNumPVSystems:
    """Reduce the number of PV systems per example to `requested_num_pv_systems`.

    Randomly select PV systems for each example. If there are less PV systems available
    than requested, then randomly sample with duplicates allowed.

    This is implemented as an xr_batch_processor so it can run after
    SelectPVSystemsNearCenterOfImage.
    """

    requested_num_pv_systems: int

    def __post_init__(self):
        self.rng = np.random.default_rng()  # Seeded by seed_rngs worker_init_function

    def __call__(self, xr_batch: XarrayBatch) -> XarrayBatch:
        pv_batch = xr_batch[PV]
        num_examples = len(pv_batch.example)

        selection = np.zeros(shape=(num_examples, self.requested_num_pv_systems), dtype=np.int32)
        for example_i in range(num_examples):
            pv_mask_for_example = pv_batch.pv_mask.isel(example=example_i).values
            all_indicies = np.nonzero(pv_mask_for_example)[0]
            # Only allow a PV system to be chosen multiple times for this example if there are
            # less available PV systems than requested PV systems.
            replace = len(all_indicies) < self.requested_num_pv_systems
            chosen_indicies = self.rng.choice(
                all_indicies, size=self.requested_num_pv_systems, replace=replace
            )
            selection[example_i] = chosen_indicies

        selection = xr.DataArray(selection, dims=("example", "pv_system"))
        pv_batch = pv_batch.isel(pv_system=selection)
        xr_batch[PV] = pv_batch
        return xr_batch
