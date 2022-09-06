from typing import Callable, Iterable

import numpy as np
import pytest
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.consts import PV_SYSTEM_AXIS, PV_TIME_AXIS
from power_perceiver.load_prepared_batches.data_sources import (
    GSP,
    PV,
    HRVSatellite,
    PreparedDataSource,
)
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.load_prepared_batches.prepared_dataset import PreparedDataset
from power_perceiver.np_batch_processor import EncodeSpaceTime
from power_perceiver.testing import (
    INDEXES_OF_PUBLICLY_AVAILABLE_BATCHES_FOR_TESTING,
    download_batches_for_data_source_if_necessary,
    get_path_of_local_data_for_testing,
)
from power_perceiver.transforms.pv import PVPowerRollingWindow
from power_perceiver.transforms.satellite import PatchSatellite
from power_perceiver.xr_batch_processor import SelectPVSystemsNearCenterOfImage

_DATA_SOURCES_TO_DOWNLOAD = (HRVSatellite.name, PV.name, GSP.name)
BATCH_SIZE = 32
N_PV_TIMESTEPS = 31
N_PV_SYSTEMS_PER_EXAMPLE = 128


def setup_module():
    for data_source_name in _DATA_SOURCES_TO_DOWNLOAD:
        download_batches_for_data_source_if_necessary(data_source_name)


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
@pytest.mark.parametrize(
    argnames=["max_n_batches_per_epoch", "expected_n_batches"],
    argvalues=[(None, len(INDEXES_OF_PUBLICLY_AVAILABLE_BATCHES_FOR_TESTING)), (1, 1)],
)
def test_init(max_n_batches_per_epoch: int, expected_n_batches: int):
    dataset = PreparedDataset(
        data_path=get_path_of_local_data_for_testing(),
        data_loaders=[PV()],
        max_n_batches_per_epoch=max_n_batches_per_epoch,
    )
    assert dataset.n_batches == expected_n_batches
    assert len(dataset) == expected_n_batches


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
@pytest.mark.parametrize(
    argnames=["data_loader", "expected_batch_keys"],
    argvalues=[
        (HRVSatellite(), [BatchKey.hrvsatellite_actual]),
        (PV(), [BatchKey.pv, BatchKey.pv_system_row_number]),
    ],
)
def test_dataset_with_single_data_source(
    data_loader: PreparedDataSource, expected_batch_keys: Iterable[BatchKey]
):
    dataset = PreparedDataset(
        data_path=get_path_of_local_data_for_testing(),
        data_loaders=[data_loader],
    )
    np_batch = dataset[0]
    assert len(np_batch) > 0
    assert isinstance(np_batch, dict)
    assert all([isinstance(key, BatchKey) for key in np_batch])
    assert all([isinstance(value, np.ndarray) for value in np_batch.values()])
    for batch_key in expected_batch_keys:
        assert batch_key in np_batch, f"{batch_key} not in np_data!"


def _check_pv_batch(
    np_batch: NumpyBatch,
    expected_batch_size: int = BATCH_SIZE,
    expected_n_pv_timesteps: int = N_PV_TIMESTEPS,
    expected_n_pv_systems_per_example: int = N_PV_SYSTEMS_PER_EXAMPLE,
) -> None:

    pv_batch = np_batch[BatchKey.pv]
    assert len(pv_batch.shape) == 3
    assert pv_batch.shape[0] == expected_batch_size
    assert pv_batch.shape[PV_TIME_AXIS] == expected_n_pv_timesteps
    assert pv_batch.shape[PV_SYSTEM_AXIS] == expected_n_pv_systems_per_example

    pv_system_row_number = np_batch[BatchKey.pv_system_row_number]
    assert len(pv_system_row_number.shape) == 2
    assert pv_system_row_number.shape[0] == expected_batch_size
    assert pv_system_row_number.shape[1] == expected_n_pv_systems_per_example

    pv_mask = np_batch[BatchKey.pv_mask]
    assert len(pv_mask.shape) == 2
    assert pv_mask.shape[0] == expected_batch_size
    assert pv_mask.shape[1] == expected_n_pv_systems_per_example

    # Select valid PV systems, and check the timeseries are not NaN.
    assert pv_mask.any(), "No valid PV systems!"
    pv_is_finite = np.isfinite(pv_batch).all(axis=PV_TIME_AXIS)
    assert ((~pv_is_finite | ~pv_mask) == ~pv_mask).all()


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_select_pv_systems_near_center_of_image():
    xr_batch_processors = [SelectPVSystemsNearCenterOfImage()]

    dataset = PreparedDataset(
        data_path=get_path_of_local_data_for_testing(),
        data_loaders=[HRVSatellite(), PV()],
        xr_batch_processors=xr_batch_processors,
    )
    np_batch = dataset[0]
    # Batch 0 has 1 example with no PV systems within the region of interest.
    _check_pv_batch(np_batch, expected_batch_size=BATCH_SIZE - 1)

    # Batch 1 has 4 examples with no PV systems within the region of interest.
    np_batch = dataset[1]
    _check_pv_batch(np_batch, expected_batch_size=BATCH_SIZE - 4)


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
@pytest.mark.parametrize(argnames="transforms", argvalues=[None, [PVPowerRollingWindow()]])
def test_pv(transforms: Iterable[Callable]):
    pv_data_loader = PV(transforms=transforms)
    dataset = PreparedDataset(
        data_path=get_path_of_local_data_for_testing(),
        data_loaders=[pv_data_loader],
    )
    assert len(dataset) == len(INDEXES_OF_PUBLICLY_AVAILABLE_BATCHES_FOR_TESTING)
    for batch_idx in range(len(INDEXES_OF_PUBLICLY_AVAILABLE_BATCHES_FOR_TESTING)):
        np_batch = dataset[batch_idx]
        _check_pv_batch(np_batch)


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_all_data_loaders_and_all_transforms():
    dataset = PreparedDataset(
        data_path=get_path_of_local_data_for_testing(),
        data_loaders=[
            HRVSatellite(transforms=[PatchSatellite()]),
            PV(transforms=[PVPowerRollingWindow()]),
        ],
        xr_batch_processors=[SelectPVSystemsNearCenterOfImage()],
        np_batch_processors=[EncodeSpaceTime()],
    )
    for batch_idx in range(len(INDEXES_OF_PUBLICLY_AVAILABLE_BATCHES_FOR_TESTING)):
        np_batch = dataset[batch_idx]
        if batch_idx == 1:
            # Batch 1 has 4 examples with no PV systems within the region of interest.
            expected_batch_size = BATCH_SIZE - 4
        else:
            expected_batch_size = BATCH_SIZE - 1
        _check_pv_batch(np_batch, expected_batch_size=expected_batch_size)
        # shape is (example, time, channel, y, x, patch)
        assert np_batch[BatchKey.hrvsatellite_actual].shape == (
            expected_batch_size,
            31,
            1,
            16,
            16,
            16,
        )
        assert np_batch[BatchKey.hrvsatellite_x_osgb].shape == (expected_batch_size, 16, 16)
        assert np_batch[BatchKey.hrvsatellite_y_osgb].shape == (expected_batch_size, 16, 16)
        assert np_batch[BatchKey.hrvsatellite_x_osgb_fourier].shape == (
            expected_batch_size,
            16,
            16,
            8,
        )
        assert np_batch[BatchKey.hrvsatellite_y_osgb_fourier].shape == (
            expected_batch_size,
            16,
            16,
            8,
        )
