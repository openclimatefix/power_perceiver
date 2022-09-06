import datetime

import numpy as np
import pandas as pd
import pytest
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_raw.data_sources.raw_gsp_data_source import RawGSPDataSource
from power_perceiver.load_raw.data_sources.raw_satellite_data_source import RawSatelliteDataSource
from power_perceiver.load_raw.raw_dataset import RawDataset
from power_perceiver.np_batch_processor.align_gsp_to_5_min import AlignGSPTo5Min

N_EXAMPLES_PER_BATCH = 16


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_raw_dataset_with_sat_and_gsp(
    sat_data_source: RawSatelliteDataSource,
    gsp_data_source: RawGSPDataSource,
):
    dataset = RawDataset(
        data_source_combos=dict(
            sat_and_gsp=(gsp_data_source, sat_data_source),
        ),
        min_duration_to_load_per_epoch=datetime.timedelta(hours=20),
        n_batches_per_epoch=16,
        n_examples_per_batch=N_EXAMPLES_PER_BATCH,
        np_batch_processors=[AlignGSPTo5Min()],
    )
    dataset.per_worker_init(worker_id=0, seed=0)
    for np_batch in dataset:
        break

    gsp_5_min = np_batch[BatchKey.gsp_5_min]
    gsp_5_min_time_utc = np_batch[BatchKey.gsp_5_min_time_utc]
    hrv_time_utc = np_batch[BatchKey.hrvsatellite_time_utc]
    assert gsp_5_min.shape[0] == N_EXAMPLES_PER_BATCH
    assert gsp_5_min_time_utc.shape[0] == N_EXAMPLES_PER_BATCH
    assert gsp_5_min.shape[1] == hrv_time_utc.shape[1]

    for example_i in range(N_EXAMPLES_PER_BATCH):
        gsp_5_min_dt = pd.to_datetime(gsp_5_min_time_utc[example_i], unit="s")
        hrv_time_dt = pd.to_datetime(hrv_time_utc[example_i], unit="s")

        np.testing.assert_array_equal(gsp_5_min_dt, hrv_time_dt.ceil("30T"))
