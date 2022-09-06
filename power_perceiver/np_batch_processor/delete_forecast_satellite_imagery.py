from dataclasses import dataclass

from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch
from power_perceiver.utils import assert_num_dims


@dataclass
class DeleteForecastSatelliteImagery:
    """Delete imagery of the future.

    Useful when not training the U-Net, and we want to save GPU RAM.

    But we do want hrvsatellite_time_utc to continue out to 2 hours because
    downstream code relies on hrvsatellite_time_utc.
    """

    num_hist_sat_images: int

    def __call__(self, np_batch: NumpyBatch) -> NumpyBatch:
        # Shape: example, time, channels, y, x
        assert_num_dims(np_batch[BatchKey.hrvsatellite_actual], 5)
        np_batch[BatchKey.hrvsatellite_actual] = np_batch[BatchKey.hrvsatellite_actual][
            :, : self.num_hist_sat_images
        ]
        return np_batch
