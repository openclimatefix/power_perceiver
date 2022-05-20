from copy import deepcopy

import numpy as np
import pytest
from conftest import SAT_HEIGHT_IN_PIXELS, SAT_N_EXPECTED_TIMESTEPS, SAT_WIDTH_IN_PIXELS

from power_perceiver.consts import BatchKey
from power_perceiver.load_raw.raw_dataset import RawDataset


def test_init(raw_dataset_with_sat_only: RawDataset):
    assert len(raw_dataset_with_sat_only._all_t0_periods_per_combo) == 0
    assert len(raw_dataset_with_sat_only._t0_datetimes_per_combo_for_epoch) == 0


def test_per_worker_init(raw_dataset_with_sat_only: RawDataset):
    dataset = deepcopy(raw_dataset_with_sat_only)
    dataset.per_worker_init(worker_id=1)


@pytest.mark.parametrize(
    "raw_dataset_str", ["raw_dataset_with_sat_only", "raw_dataset_with_sat_only_and_gsp_pv_sat"]
)
def test_iter(raw_dataset_str: str, request):
    # `getfixturevalue` trick from https://stackoverflow.com/a/64348247/732596
    raw_dataset: RawDataset = request.getfixturevalue(raw_dataset_str)
    dataset = deepcopy(raw_dataset)
    dataset.per_worker_init(worker_id=1)
    for np_example in dataset:
        break
    for key, value in np_example.items():
        assert value.dtype.type == np.float32, f"{key.name=} has {value.dtype=}, not float32!"
        assert np.isfinite(value).all(), f"{key.name=} has non-finite values!"
        print(key, value.shape)

    for key, expected_shape in (
        (
            BatchKey.hrvsatellite,
            (SAT_N_EXPECTED_TIMESTEPS, 1, SAT_HEIGHT_IN_PIXELS, SAT_WIDTH_IN_PIXELS),
        ),
        (BatchKey.hrvsatellite_time_utc, (SAT_N_EXPECTED_TIMESTEPS,)),
        (BatchKey.hrvsatellite_y_osgb, (SAT_HEIGHT_IN_PIXELS, SAT_WIDTH_IN_PIXELS)),
        (BatchKey.hrvsatellite_x_osgb, (SAT_HEIGHT_IN_PIXELS, SAT_WIDTH_IN_PIXELS)),
        (BatchKey.hrvsatellite_y_geostationary, (SAT_HEIGHT_IN_PIXELS,)),
        (BatchKey.hrvsatellite_x_geostationary, (SAT_WIDTH_IN_PIXELS,)),
    ):
        value = np_example[key]
        assert (
            value.shape == expected_shape
        ), f"{key.name=} has shape {value.shape}, not {expected_shape}"


def test_dataset_with_sat_only_and_gsp_pv_sat(raw_dataset_with_sat_only_and_gsp_pv_sat: RawDataset):
    pass
