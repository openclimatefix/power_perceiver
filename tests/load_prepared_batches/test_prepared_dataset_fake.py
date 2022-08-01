from typing import Callable, Iterable

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from nowcasting_dataset.dataset.batch import Batch
from nowcasting_dataset.config.model import Configuration

from power_perceiver.consts import PV_SYSTEM_AXIS, PV_TIME_AXIS, BatchKey
from power_perceiver.load_prepared_batches.data_sources import (
    GSP,
    NWP,
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

_DATA_SOURCES_TO_DOWNLOAD = (HRVSatellite.name, PV.name, GSP.name, NWP.name)
BATCH_SIZE = 4
N_PV_TIMESTEPS = 19
N_PV_SYSTEMS_PER_EXAMPLE = 128


def make_batches(path: Path, number_batches: int):

    c = Configuration()
    c.input_data = c.input_data.set_all_to_defaults()
    c.process.batch_size = 4
    c.input_data.nwp.nwp_image_size_pixels_height = 10
    c.input_data.nwp.nwp_image_size_pixels_width = 10

    c.input_data.pv.n_pv_systems_per_example = N_PV_SYSTEMS_PER_EXAMPLE

    c.input_data.satellite.satellite_image_size_pixels_width = 10
    c.input_data.satellite.satellite_image_size_pixels_height = 10

    c.input_data.hrvsatellite.hrvsatellite_image_size_pixels_width = 20
    c.input_data.hrvsatellite.hrvsatellite_image_size_pixels_height = 20

    for i in range(0, number_batches):
        batch = Batch.fake(configuration=c)
        batch.save_netcdf(batch_i=i, path=path)


@pytest.mark.parametrize(
    argnames=["max_n_batches_per_epoch", "expected_n_batches"],
    argvalues=[(1, 1), (3, 3)],
)
def test_init(max_n_batches_per_epoch: int, expected_n_batches: int):

    with tempfile.TemporaryDirectory() as temp_dir:
        make_batches(path=Path(temp_dir), number_batches=max_n_batches_per_epoch)

        dataset = PreparedDataset(
            data_path=Path(temp_dir),
            data_loaders=[PV(history_duration=pd.Timedelta("90 min"))],
            max_n_batches_per_epoch=max_n_batches_per_epoch,
        )
        assert dataset.n_batches == expected_n_batches
        assert len(dataset) == expected_n_batches


@pytest.mark.parametrize(
    argnames=["data_loader", "expected_batch_keys"],
    argvalues=[
        (HRVSatellite(history_duration=pd.Timedelta("30 min")), [BatchKey.hrvsatellite_actual]),
        (PV(history_duration=pd.Timedelta("90 min")), [BatchKey.pv, BatchKey.pv_system_row_number]),
    ],
)
def test_dataset_with_single_data_source(
    data_loader: PreparedDataSource, expected_batch_keys: Iterable[BatchKey]
):

    with tempfile.TemporaryDirectory() as temp_dir:
        make_batches(path=Path(temp_dir), number_batches=3)

        dataset = PreparedDataset(
            data_path=Path(temp_dir),
            data_loaders=[data_loader],
        )
        np_batch = dataset[0]
        assert len(np_batch) > 0
        assert isinstance(np_batch, dict)
        assert all([isinstance(key, BatchKey) for key in np_batch])
        assert all(
            [
                isinstance(value, np.ndarray)
                for key, value in np_batch.items()
                if "t0_idx" not in key.name
            ]
        )
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


def test_select_pv_systems_near_center_of_image():
    xr_batch_processors = [SelectPVSystemsNearCenterOfImage(drop_examples=False)]

    with tempfile.TemporaryDirectory() as temp_dir:
        make_batches(path=Path(temp_dir), number_batches=2)

        dataset = PreparedDataset(
            data_path=Path(temp_dir),
            data_loaders=[
                HRVSatellite(history_duration=pd.Timedelta("30 min")),
                PV(history_duration=pd.Timedelta("30 min")),
            ],
            xr_batch_processors=xr_batch_processors,
        )
        np_batch = dataset[0]
        # Batch 0 has 4 example with no PV systems within the region of interest.
        _check_pv_batch(np_batch, expected_batch_size=BATCH_SIZE)

        # Batch 1 has 4 examples with no PV systems within the region of interest.
        np_batch = dataset[1]
        _check_pv_batch(np_batch, expected_batch_size=BATCH_SIZE)


@pytest.mark.parametrize(argnames="transforms", argvalues=[None, [PVPowerRollingWindow()]])
def test_pv(transforms: Iterable[Callable]):
    pv_data_loader = PV(transforms=transforms, history_duration=pd.Timedelta("30 min"))

    with tempfile.TemporaryDirectory() as temp_dir:
        make_batches(path=Path(temp_dir), number_batches=2)

        dataset = PreparedDataset(
            data_path=Path(temp_dir),
            data_loaders=[pv_data_loader],
        )
        assert len(dataset) == len(INDEXES_OF_PUBLICLY_AVAILABLE_BATCHES_FOR_TESTING)
        for batch_idx in range(len(INDEXES_OF_PUBLICLY_AVAILABLE_BATCHES_FOR_TESTING)):
            np_batch = dataset[batch_idx]
            _check_pv_batch(np_batch)


def test_all_data_loaders_and_all_transforms():
    with tempfile.TemporaryDirectory() as temp_dir:
        make_batches(path=Path(temp_dir), number_batches=2)

        dataset = PreparedDataset(
            data_path=Path(temp_dir),
            data_loaders=[
                HRVSatellite(transforms=[PatchSatellite()], history_duration=pd.Timedelta("30 min")),
                PV(transforms=[PVPowerRollingWindow()], history_duration=pd.Timedelta("90 min")),
            ],
            xr_batch_processors=[SelectPVSystemsNearCenterOfImage()],
            np_batch_processors=[EncodeSpaceTime()],
        )
        for batch_idx in range(len(INDEXES_OF_PUBLICLY_AVAILABLE_BATCHES_FOR_TESTING)):
            np_batch = dataset[batch_idx]

            _check_pv_batch(np_batch, expected_batch_size=BATCH_SIZE)
            # shape is (example, time, channel, y, x, patch)
            assert np_batch[BatchKey.hrvsatellite_actual].shape == (
                BATCH_SIZE,
                N_PV_TIMESTEPS,
                1,
                5,
                5,
                16,
            )
            assert np_batch[BatchKey.hrvsatellite_x_osgb].shape == (BATCH_SIZE, 5, 5)
            assert np_batch[BatchKey.hrvsatellite_y_osgb].shape == (BATCH_SIZE, 5, 5)
            assert np_batch[BatchKey.hrvsatellite_x_osgb_fourier].shape == (
                BATCH_SIZE,
                5,
                5,
                8,
            )
            assert np_batch[BatchKey.hrvsatellite_y_osgb_fourier].shape == (
                BATCH_SIZE,
                5,
                5,
                8,
            )
