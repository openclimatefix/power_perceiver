import numpy as np
import pytest
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_prepared_batches.data_sources import PV, HRVSatellite
from power_perceiver.load_prepared_batches.prepared_dataset import PreparedDataset
from power_perceiver.np_batch_processor import EncodeSpaceTime
from power_perceiver.np_batch_processor.encode_space_time import compute_fourier_features
from power_perceiver.testing import get_path_of_local_data_for_testing


def test_fourier_features():
    test_array = np.arange(start=-2, stop=2, step=0.01, dtype=np.float32).reshape((1, -1))
    fourier_features = compute_fourier_features(test_array)
    assert fourier_features.shape == (1, 400, 8)
    assert fourier_features.dtype == np.float32


@pytest.mark.skip(
    "Skip for the moment - https://github.com/openclimatefix/power_perceiver/issues/187"
)
def test_encode_space_time():
    dataset = PreparedDataset(
        data_path=get_path_of_local_data_for_testing(),
        data_loaders=[HRVSatellite(), PV()],
        np_batch_processors=[EncodeSpaceTime()],
    )
    np_batch = dataset[0]

    for batch_key in (
        BatchKey.hrvsatellite_time_utc_fourier,
        BatchKey.hrvsatellite_x_osgb_fourier,
        BatchKey.hrvsatellite_y_osgb_fourier,
        BatchKey.pv_time_utc_fourier,
        BatchKey.pv_x_osgb_fourier,
        BatchKey.pv_y_osgb_fourier,
    ):
        assert batch_key in np_batch, f"{batch_key.name} is missing from np_batch!"

    assert np_batch[BatchKey.hrvsatellite_x_osgb_fourier].shape == (32, 64, 64, 8)
