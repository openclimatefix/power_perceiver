from typing import Iterable

import numpy as np
import pytest

from power_perceiver.consts import BatchKey, DataSourceName
from power_perceiver.dataset import NowcastingDataset
from power_perceiver.testing import (
    INDEXES_OF_PUBLICLY_AVAILABLE_BATCHES_FOR_TESTING,
    download_batches_for_data_source_if_necessary,
    get_path_of_local_data_for_testing,
)

_DATA_SOURCES_TO_DOWNLOAD = (
    DataSourceName.satellite,
    DataSourceName.hrvsatellite,
    DataSourceName.pv,
    DataSourceName.nwp,
)


def setup_module():
    for data_source_name in _DATA_SOURCES_TO_DOWNLOAD:
        download_batches_for_data_source_if_necessary(data_source_name)


@pytest.mark.parametrize(
    argnames=["max_n_batches_per_epoch", "expected_n_batches"],
    argvalues=[(None, len(INDEXES_OF_PUBLICLY_AVAILABLE_BATCHES_FOR_TESTING)), (1, 1)],
)
def test_init(max_n_batches_per_epoch: int, expected_n_batches: int):
    dataset = NowcastingDataset(
        data_path=get_path_of_local_data_for_testing(),
        data_source_names=_DATA_SOURCES_TO_DOWNLOAD,
        max_n_batches_per_epoch=max_n_batches_per_epoch,
    )
    assert dataset.n_batches == expected_n_batches
    assert len(dataset) == expected_n_batches


@pytest.mark.parametrize(
    argnames=["data_source_name", "expected_batch_keys"],
    argvalues=[
        (DataSourceName.hrvsatellite, [BatchKey.hrvsatellite]),
        (DataSourceName.pv, [BatchKey.pv, BatchKey.pv_system_row_number]),
    ],
)
def test_dataset_with_single_data_source(
    data_source_name: DataSourceName, expected_batch_keys: Iterable[BatchKey]
):
    dataset = NowcastingDataset(
        data_path=get_path_of_local_data_for_testing(),
        data_source_names=[data_source_name],
    )
    np_data = dataset[0]
    assert len(np_data) > 0
    assert isinstance(np_data, dict)
    assert all([isinstance(key, BatchKey) for key in np_data])
    assert all([isinstance(value, np.ndarray) for value in np_data.values()])
    for batch_key in expected_batch_keys:
        assert batch_key in np_data, f"{batch_key} not in np_data!"
