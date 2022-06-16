from dataclasses import dataclass

from power_perceiver.consts import BatchKey
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch


@dataclass
class DeleteForecastSatelliteImagery:
    """Delete imagery of the future.

    Useful when not training the U-Net, and we want to save GPU RAM.

    But we do want hrvsatellite_time_utc to continue out to 2 hours because
    downstream code relies on hrvsatellite_time_utc.
    """

    num_hist_sat_images: int

    def __call__(self, np_batch: NumpyBatch) -> NumpyBatch:
        # Shape: time, channels, y, x
        np_batch[BatchKey.hrvsatellite] = np_batch[BatchKey.hrvsatellite][
            : self.num_hist_sat_images
        ]
        return np_batch
