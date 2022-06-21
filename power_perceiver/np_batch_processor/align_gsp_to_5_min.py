from dataclasses import dataclass

import numpy as np
import pandas as pd

from power_perceiver.consts import BatchKey
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.utils import datetime64_to_float, stack_np_examples_into_batch


@dataclass
class AlignGSPTo5Min:
    """Aligns GSP data to 5 min data.

    The GSP data isn't interpolated. Instead, for each 5_min_timestep, we take the GSP data at
    5_min_timestep.ceil("30T"). If that GSP timestep does not exist then NaNs will be used.
    """

    batch_key_for_5_min_datetimes: BatchKey = BatchKey.hrvsatellite_time_utc

    def __call__(self, np_batch: NumpyBatch) -> NumpyBatch:
        # Loop through each example and find the index into the GSP time dimension
        # of the GSP timestep corresponding to each 5 minute timestep:
        gsp_5_min_for_all_examples: list[NumpyBatch] = []
        n_examples = np_batch[BatchKey.gsp].shape[0]
        for example_i in range(n_examples):
            # Find the corresponding GSP 30 minute timestep for each 5 minute satellite timestep.
            # We do this by taking the `ceil("30T")` of each 5 minute satellite timestep.
            # Most of the code below is just converting to Pandas and back
            # so we can use `pd.DatetimeIndex.ceil` on each datetime:
            time_5_min = np_batch[self.batch_key_for_5_min_datetimes][example_i]
            time_5_min_dt_index = pd.to_datetime(time_5_min, unit="s")
            time_30_min_every_5_min_dt_index = time_5_min_dt_index.ceil("30T")
            time_30_min_every_5_min = datetime64_to_float(time_30_min_every_5_min_dt_index.values)

            # Now, find the index into the original 30-minute GSP data for each 5-min timestep:
            gsp_30_min_time = np_batch[BatchKey.gsp_time_utc][example_i]
            idx_into_gsp = np.searchsorted(gsp_30_min_time, time_30_min_every_5_min)

            gsp_5_min_example: NumpyBatch = {}
            for batch_key in (BatchKey.gsp, BatchKey.gsp_time_utc):
                new_batch_key_name = batch_key.name.replace("gsp", "gsp_5_min")
                new_batch_key = BatchKey[new_batch_key_name]
                gsp_5_min_example[new_batch_key] = np_batch[batch_key][example_i, idx_into_gsp]

            gsp_5_min_for_all_examples.append(gsp_5_min_example)

        # Stack the individual examples back into a batch of examples:
        new_np_batch = stack_np_examples_into_batch(gsp_5_min_for_all_examples)
        np_batch.update(new_np_batch)

        # Copy over the t0_idx scalar:
        batch_key_name_for_5_min_t0_idx = self.batch_key_for_5_min_datetimes.name.replace(
            "time_utc", "t0_idx"
        )
        batch_key_for_5_min_t0_idx = BatchKey[batch_key_name_for_5_min_t0_idx]
        np_batch[BatchKey.gsp_5_min_t0_idx] = np_batch[batch_key_for_5_min_t0_idx]

        return np_batch
