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


class PV(PreparedDataSource):
    def process_before_transforms(self, dataset: xr.Dataset) -> xr.Dataset:
        # None of this will be necessary once this is implemented:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/630

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
                "power_mw": "power_w",
                "capacity_mwp": "capacity_wp",
                "id": "pv_system_id",
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
                "id_index": "pv_system",
            }
        )

        # Setting coords won't be necessary once this is fixed:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/627
        dataset = dataset.set_coords(
            ["time_utc", "pv_system_id", "pv_system_row_number", "y_osgb", "x_osgb"]
        )

        dataset = dataset.transpose("example", "time", "pv_system")

        # PV power
        # Note that some capacities are 0. This will be fixed upstream in:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/622
        dataset["power_normalised"] = dataset["power_w"] / dataset["capacity_wp"]
        dataset["power_normalised"] = dataset["power_normalised"].astype(np.float32)

        # Compute mask of valid PV data.
        # The mask will be a bool DataArray of shape (batch_size, n_pv_systems).
        # In v15, some capacities are 0. This will be fixed upstream in:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/622
        valid_pv_capacity = dataset["capacity_wp"] > 0
        # A NaN ID value is the "official" way to indicate a missing or deselected PV system.
        valid_pv_id = np.isfinite(dataset["pv_system_id"])
        valid_pv_power = np.isfinite(dataset["power_normalised"]).all(dim="time")
        pv_mask = valid_pv_capacity & valid_pv_id & valid_pv_power
        assert pv_mask.any(), "No valid PV systems!"
        dataset["pv_mask"] = pv_mask

        # Apply the mask. In particular, this is especially important so we set the
        # osgb coordinates to NaN for missing PV systems. In v15, the osgb coordinates
        # are zero for missing PV systems, which confuses `EncodeSpaceTime`!
        dataset = apply_pv_mask(dataset)

        # PV spatial coords are float64 in v15. This will be fixed in:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/624
        for dataset_key in ("x_osgb", "y_osgb"):
            dataset[dataset_key] = dataset[dataset_key].astype(np.float32)

        return dataset

    @staticmethod
    def to_numpy(dataset: xr.Dataset) -> NumpyBatch:
        """This is called from Dataset.__getitem__.

        This processes this modality's xr.Dataset, to convert the xr.Dataset
        into a dictionary mapping BatchKeys to numpy arrays, as documented
        in the BatchKey class.
        """
        batch: NumpyBatch = {}

        # PV power
        # Note that some capacities are 0. This will be fixed upstream in:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/622
        batch[BatchKey.pv] = dataset["power_normalised"].values

        # In v15 of the dataset, `pv_system_row_number` is int64. This will be fixed in:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/624
        batch[BatchKey.pv_system_row_number] = dataset["pv_system_row_number"].values.astype(
            np.int32
        )

        # id is float64 in v15 of the dataset. This will be fixed upstream in:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/624
        batch[BatchKey.pv_id] = dataset["pv_system_id"].values.astype(np.float32)
        batch[BatchKey.pv_capacity_wp] = dataset["capacity_wp"].values

        batch[BatchKey.pv_mask] = dataset["pv_mask"].values

        # Coordinates
        batch[BatchKey.pv_time_utc] = datetime64_to_float(dataset["time_utc"].values)
        for batch_key, dataset_key in (
            (BatchKey.pv_x_osgb, "x_osgb"),
            (BatchKey.pv_y_osgb, "y_osgb"),
        ):
            batch[batch_key] = dataset[dataset_key].values

        return batch


def apply_pv_mask(dataset: xr.Dataset) -> xr.Dataset:
    """Apply mask to whole dataset (to set everything masked to NaN)"""
    # Coords for missing PV systems are set to 0 in v15. This is dangerous! So set to NaN.
    # We need to set them to NaN so `np.nanmax()` does the right thing in `EncodeSpaceTime`
    # This won't be necessary after this issue is closed:
    # https://github.com/openclimatefix/nowcasting_dataset/issues/625

    dataset = dataset.where(dataset.pv_mask)
    # `Dataset.where` sets `pv_mask` to zero. We want `pv_mask` to be a bool array:
    dataset["pv_mask"] = dataset.pv_mask.fillna(0).astype(bool)
    # `Dataset.where` *only* operates on data variables, so we need to manually mask the coords:
    for coord_name in ["pv_system_id", "x_osgb", "y_osgb"]:
        dataset[coord_name] = dataset[coord_name].where(dataset.pv_mask)
    return dataset
