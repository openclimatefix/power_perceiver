import numpy as np
import pandas as pd
import xarray as xr

from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import (
    BatchKey,
    NumpyBatch,
    PreparedDataSource,
)
from power_perceiver.utils import datetime64_to_float

# Means and std computed with
# nowcasting_dataset/scripts/compute_stats_from_batches.py
# using v15 training batches on 2021-11-24.
NWP_MEAN = {
    "t": 285.7799539185846,
    "dswrf": 294.6696933986283,
    "prate": 3.6078121378638696e-05,
    "r": 75.57106712435926,
    "sde": 0.0024915961594965614,
    "si10": 4.931356852411006,
    "vis": 22321.762918384553,
    "lcc": 47.90454236572895,
    "mcc": 44.22781694449808,
    "hcc": 32.87577371914454,
}

NWP_STD = {
    "t": 5.017000766747606,
    "dswrf": 233.1834250473355,
    "prate": 0.00021690701537950742,
    "r": 15.705370079694358,
    "sde": 0.07560040052148084,
    "si10": 2.664583614352396,
    "vis": 12963.802514945439,
    "lcc": 40.06675870700349,
    "mcc": 41.927221148316384,
    "hcc": 39.05157559763763,
}

NWP_CHANNEL_NAMES = tuple(NWP_STD.keys())


def _to_data_array(d):
    return xr.DataArray(
        [d[key] for key in NWP_CHANNEL_NAMES], coords={"channel": list(NWP_CHANNEL_NAMES)}
    ).astype(np.float32)


NWP_MEAN = _to_data_array(NWP_MEAN)
NWP_STD = _to_data_array(NWP_STD)


class NWP(PreparedDataSource):

    sample_period_duration = pd.Timedelta("1 hour")

    def process_before_transforms(self, dataset: xr.Dataset) -> xr.Dataset:
        target_times = pd.date_range(
            dataset.init_time.values,
            dataset.init_time.values + dataset.step.values[-1],
            freq=self.sample_period_duration,
        )
        dataset.coords["target_time_utc"] = target_times
        # None of this will be necessary once this is implemented:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/629

        # Downsample spatially:
        dataset = dataset.coarsen(
            y_osgb_index=16,
            x_osgb_index=16,  # Downsamples from 64x64 to 4x4
            boundary="trim",
        ).mean()

        # Rename coords to be more explicit about exactly what some coordinates hold:
        dataset = dataset.rename_vars(
            {
                "channels": "channel_name",
                "time": "time_utc",
            }
        )

        # Rename dimensions. Standardize on the singular (time, channel, etc.).
        # Remove redundant "index" from the dim name. These are *dimensions* so,
        # by definition, they are indicies!
        dataset = dataset.rename_dims(
            {
                "y_osgb_index": "y",
                "x_osgb_index": "x",
                "time_index": "time",
                "channels_index": "channel",
            }
        )
        dataset.attrs["t0_idx"] = self.t0_idx
        return dataset

    @staticmethod
    def to_numpy(dataset: xr.Dataset) -> NumpyBatch:
        """This is called from Dataset.__getitem__.

        This processes this modality's xr.Dataset, to convert the xr.Dataset
        into a dictionary mapping BatchKeys to numpy arrays, as documented
        in the BatchKey class.
        """
        example: NumpyBatch = {}

        example[BatchKey.nwp] = dataset.data.values
        example[BatchKey.nwp_t0_idx] = dataset.attrs["t0_idx"]
        target_time = dataset.coords["target_time_utc"].values
        # Need to copy batch_size times I think TODO Check this is right?
        target_time = np.repeat(np.expand_dims(target_time, axis=0), 32, axis=0)
        example[BatchKey.nwp_target_time_utc] = datetime64_to_float(target_time)
        example[BatchKey.nwp_channel_names] = dataset.channel.values
        example[BatchKey.nwp_step] = (dataset.step.values / np.timedelta64(1, "h")).astype(np.int64)
        # TODO Ensure this is right
        init_times = np.repeat(np.expand_dims(dataset.init_time.values, axis=0), 32, axis=0)
        example[BatchKey.nwp_init_time_utc] = datetime64_to_float(init_times)

        for batch_key, dataset_key in (
            (BatchKey.nwp_y_osgb, "y_osgb"),
            (BatchKey.nwp_x_osgb, "x_osgb"),
        ):
            example[batch_key] = dataset[dataset_key].values

        return example
