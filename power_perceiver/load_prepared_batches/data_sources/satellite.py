import numpy as np
import xarray as xr
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import (
    NumpyBatch,
    PreparedDataSource,
)
from power_perceiver.utils import datetime64_to_float

SAT_MEAN = {
    "HRV": 236.13257536395903,
    "IR_016": 291.61620182554185,
    "IR_039": 858.8040610176552,
    "IR_087": 738.3103442750336,
    "IR_097": 773.0910794778366,
    "IR_108": 607.5318145165666,
    "IR_120": 860.6716261423857,
    "IR_134": 925.0477987594331,
    "VIS006": 228.02134593063957,
    "VIS008": 257.56333202381205,
    "WV_062": 633.5975770915588,
    "WV_073": 543.4963868823854,
}

SAT_STD = {
    "HRV": 935.9717382401759,
    "IR_016": 172.01044433112992,
    "IR_039": 96.53756504807913,
    "IR_087": 96.21369354283686,
    "IR_097": 86.72892737648276,
    "IR_108": 156.20651744208888,
    "IR_120": 104.35287930753246,
    "IR_134": 104.36462050405994,
    "VIS006": 150.2399269307514,
    "VIS008": 152.16086321818398,
    "WV_062": 111.8514878214775,
    "WV_073": 106.8855172848904,
}


SATELLITE_CHANNEL_ORDER = ("example", "time", "channel", "y", "x")


def _set_sat_coords(dataset: xr.Dataset) -> xr.Dataset:
    """Set variables as coordinates"""
    return dataset.set_coords(
        ["time_utc", "channel_name", "y_osgb", "x_osgb", "y_geostationary", "x_geostationary"]
    )


class HRVSatellite(PreparedDataSource):
    def process_before_transforms(self, dataset: xr.Dataset) -> xr.Dataset:
        # None of this will be necessary once this is implemented:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/629

        # Drop redundant coordinates (these are redundant because they
        # just repeat the contents of each *dimension*):
        dataset = dataset.drop_vars(
            [
                "example",
                "y_geostationary_index",
                "x_geostationary_index",
                "time_index",
                "channels_index",
            ]
        )

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
                "y_geostationary_index": "y",
                "x_geostationary_index": "x",
                "time_index": "time",
                "channels_index": "channel",
            }
        )

        # Setting coords won't be necessary once this is fixed:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/627
        dataset = _set_sat_coords(dataset)

        dataset = dataset.transpose(*SATELLITE_CHANNEL_ORDER)

        return dataset

    @staticmethod
    def to_numpy(dataset: xr.Dataset) -> NumpyBatch:
        """This is called from Dataset.__getitem__.

        This processes this modality's xr.Dataset, to convert the xr.Dataset
        into a dictionary mapping BatchKeys to numpy arrays, as documented
        in the BatchKey class.
        """
        batch: NumpyBatch = {}

        # Prepare the satellite imagery itself
        hrvsatellite = dataset["data"]
        # hrvsatellite is int16 on disk
        hrvsatellite = hrvsatellite.astype(np.float32)
        hrvsatellite -= SAT_MEAN["HRV"]
        hrvsatellite /= SAT_STD["HRV"]
        batch[BatchKey.hrvsatellite_actual] = hrvsatellite.values

        # Coordinates
        batch[BatchKey.hrvsatellite_time_utc] = datetime64_to_float(dataset["time_utc"].values)
        for batch_key, dataset_key in (
            (BatchKey.hrvsatellite_y_osgb, "y_osgb"),
            (BatchKey.hrvsatellite_x_osgb, "x_osgb"),
            (BatchKey.hrvsatellite_y_geostationary, "y_geostationary"),
            (BatchKey.hrvsatellite_x_geostationary, "x_geostationary"),
        ):
            # HRVSatellite coords are already float32.
            batch[batch_key] = dataset[dataset_key].values

        return batch
