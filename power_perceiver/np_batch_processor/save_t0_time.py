from dataclasses import dataclass

from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import NumpyBatch


@dataclass
class SaveT0Time:
    """Save the T0 time_utc_fourier for PV and GSP data.

    This is useful to ensure the model always knows the t0 timestep,
    even after the timesteps have been subsampled.
    """

    def __call__(self, np_batch: NumpyBatch) -> NumpyBatch:
        pv_t0_idx = np_batch[BatchKey.pv_t0_idx]
        gsp_t0_idx = np_batch[BatchKey.gsp_t0_idx]
        hrvsatellite_t0_idx = np_batch[BatchKey.hrvsatellite_t0_idx]

        np_batch[BatchKey.pv_time_utc_fourier_t0] = np_batch[BatchKey.pv_time_utc_fourier][
            :, pv_t0_idx
        ]
        np_batch[BatchKey.gsp_time_utc_fourier_t0] = np_batch[BatchKey.gsp_time_utc_fourier][
            :, gsp_t0_idx
        ]
        np_batch[BatchKey.hrvsatellite_time_utc_fourier_t0] = np_batch[
            BatchKey.hrvsatellite_time_utc_fourier
        ][:, hrvsatellite_t0_idx]
        return np_batch
