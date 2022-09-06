from copy import deepcopy

import pytest
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_raw.raw_dataset import RawDataset

N_EXAMPLES_PER_BATCH = 16
SAT_HEIGHT_IN_PIXELS = 128
SAT_WIDTH_IN_PIXELS = 256
SAT_N_EXPECTED_TIMESTEPS = 37  # 12 steps of history + 1 for t0 + 24 of forecast


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_init(raw_dataset_with_sat_only: RawDataset):
    assert len(raw_dataset_with_sat_only._all_t0_periods_per_combo) == 0
    assert len(raw_dataset_with_sat_only._t0_datetimes_per_combo_for_epoch) == 0


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_per_worker_init(raw_dataset_with_sat_only: RawDataset):
    dataset = deepcopy(raw_dataset_with_sat_only)
    dataset.per_worker_init(worker_id=1)


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
@pytest.mark.parametrize(
    "raw_dataset_str", ["raw_dataset_with_sat_only", "raw_dataset_with_sat_only_and_gsp_pv_sat"]
)
def test_iter(raw_dataset_str: str, request):
    # `getfixturevalue` trick from https://stackoverflow.com/a/64348247/732596
    raw_dataset: RawDataset = request.getfixturevalue(raw_dataset_str)
    dataset = deepcopy(raw_dataset)
    dataset.per_worker_init(worker_id=1)
    for np_batch in dataset:
        break

    for key, expected_shape in (
        (
            BatchKey.hrvsatellite_actual,
            (
                N_EXAMPLES_PER_BATCH,
                SAT_N_EXPECTED_TIMESTEPS,
                1,
                SAT_HEIGHT_IN_PIXELS,
                SAT_WIDTH_IN_PIXELS,
            ),
        ),
        (
            BatchKey.hrvsatellite_time_utc,
            (
                N_EXAMPLES_PER_BATCH,
                SAT_N_EXPECTED_TIMESTEPS,
            ),
        ),
        (
            BatchKey.hrvsatellite_y_osgb,
            (N_EXAMPLES_PER_BATCH, SAT_HEIGHT_IN_PIXELS, SAT_WIDTH_IN_PIXELS),
        ),
        (
            BatchKey.hrvsatellite_x_osgb,
            (N_EXAMPLES_PER_BATCH, SAT_HEIGHT_IN_PIXELS, SAT_WIDTH_IN_PIXELS),
        ),
        (
            BatchKey.hrvsatellite_y_geostationary,
            (
                N_EXAMPLES_PER_BATCH,
                SAT_HEIGHT_IN_PIXELS,
            ),
        ),
        (
            BatchKey.hrvsatellite_x_geostationary,
            (
                N_EXAMPLES_PER_BATCH,
                SAT_WIDTH_IN_PIXELS,
            ),
        ),
    ):
        value = np_batch[key]
        assert (
            value.shape == expected_shape
        ), f"{key.name=} has shape {value.shape}, not {expected_shape}"


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_dataset_with_sat_only_and_gsp_pv_sat(raw_dataset_with_sat_only_and_gsp_pv_sat: RawDataset):
    pass
