import numpy as np
import xarray as xr

from power_perceiver.data_loader.data_loader import BatchKey, DataLoader, NumpyBatch
from power_perceiver.utils import datetime64_to_int

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


SATELLITE_CHANNEL_ORDER = (
    "example",
    "time_index",
    "channels_index",
    "y_geostationary_index",
    "x_geostationary_index",
)


class HRVSatellite(DataLoader):
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
        hrvsatellite = hrvsatellite.transpose(*SATELLITE_CHANNEL_ORDER)
        batch[BatchKey.hrvsatellite] = hrvsatellite.values

        # Coordinates
        batch[BatchKey.hrvsatellite_time_utc] = datetime64_to_int(dataset["time"].values)
        for batch_key, dataset_key in (
            (BatchKey.hrvsatellite_x_osgb, "x_osgb"),
            (BatchKey.hrvsatellite_y_osgb, "y_osgb"),
        ):
            coords = dataset[dataset_key]
            # HRVSatellite coords are already float32.
            batch[batch_key] = coords.values

        return batch
