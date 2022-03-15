import pytest

from power_perceiver.consts import DataSourceName
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


def test_dataset_with_hrvsatellite():
    dataset = NowcastingDataset(
        data_path=get_path_of_local_data_for_testing(),
        data_source_names=[DataSourceName.hrvsatellite],
    )
    dataset[0]  # TODO: Do something with this!
