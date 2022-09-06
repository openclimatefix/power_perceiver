import logging

import numpy as np
import xarray as xr
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import (
    NumpyBatch,
    PreparedDataSource,
)
from power_perceiver.utils import datetime64_to_float

_log = logging.getLogger(__name__)


class GSP(PreparedDataSource):
    def process_before_transforms(self, dataset: xr.Dataset) -> xr.Dataset:
        # None of this will be necessary once this is implemented:
        # TODO: MAKE ISSUE!

        # Drop redundant coordinates (these are redundant because they
        # just repeat the contents of each *dimension*):
        dataset = dataset.drop_vars(["example", "id_index", "time_index"])

        # Rename coords to be more explicit about exactly what some coordinates hold:
        # Note that, in v15 of the dataset, the keys are incorrectly named
        # power_mw and capacity_mwp, even though the power and capacity are both in watts.
        # See https://github.com/openclimatefix/nowcasting_dataset/issues/530
        dataset = dataset.rename_vars(
            {
                "time": "time_utc",
                "id": "gsp_id",
                "y_coords": "y_osgb",
                "x_coords": "x_osgb",
            }
        )

        # Rename dimensions. Standardize on the singular (time, channel, etc.).
        # Remove redundant "index" from the dim name. These are *dimensions* so,
        # by definition, they are indicies!
        dataset = dataset.rename_dims(
            {
                "time_index": "time",
                "id_index": "gsp",
            }
        )

        # Setting coords won't be necessary once this is fixed:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/627
        dataset = dataset.set_coords(["time_utc", "gsp_id", "y_osgb", "x_osgb"])

        # We're only interested in the target GSP!
        dataset = dataset.isel(gsp=0)

        # Most coords and variables are float64 in v15. This will be fixed in:
        # TODO: CREATE ISSUE
        for dataset_key in ("y_osgb", "x_osgb", "capacity_mwp", "power_mw"):
            dataset[dataset_key] = dataset[dataset_key].astype(np.float32)

        # GSP ID
        assert not np.isnan(dataset["gsp_id"]).any()
        dataset["gsp_id"] = dataset["gsp_id"].astype(np.int32)

        # Set OSGB coords to NaN! OSGB coords are 0 for missing GSPs in v15. This
        # confuses `EncodeSpaceTime`! So we must set missing OSGB coords to NaN.
        # This will be fixed in: TODO: CREATE ISSUE
        for dataset_key in ("y_osgb", "x_osgb"):
            data_array = dataset[dataset_key]
            dataset[dataset_key] = data_array.where(data_array != 0)

        # PV power
        # Note that some GSPs have a capacity of zero, so power_normalised will sometimes be NaN.
        dataset["power_normalised"] = dataset["power_mw"] / dataset["capacity_mwp"]

        return dataset

    @staticmethod
    def to_numpy(dataset: xr.Dataset) -> NumpyBatch:
        """This is called from Dataset.__getitem__.

        This processes this modality's xr.Dataset, to convert the xr.Dataset
        into a dictionary mapping BatchKeys to numpy arrays, as documented
        in the BatchKey class.
        """
        batch: NumpyBatch = {}

        # GSP power. Note that some capacities are 0 so some normalised power values are NaN.
        batch[BatchKey.gsp] = dataset["power_normalised"].values

        batch[BatchKey.gsp_id] = dataset["gsp_id"].values

        # Coordinates
        batch[BatchKey.gsp_time_utc] = datetime64_to_float(dataset["time_utc"].values)
        for batch_key, dataset_key in (
            (BatchKey.gsp_y_osgb, "y_osgb"),
            (BatchKey.gsp_x_osgb, "x_osgb"),
        ):
            values = dataset[dataset_key].values
            # Expand dims so EncodeSpaceTime works!
            batch[batch_key] = np.expand_dims(values, axis=1)

        # Sanity check!
        for key in (
            BatchKey.gsp,
            BatchKey.gsp_id,
            BatchKey.gsp_time_utc,
            BatchKey.gsp_y_osgb,
            BatchKey.gsp_x_osgb,
        ):
            assert not np.isnan(batch[key]).any(), f"{key} has NaNs!"

        return batch
