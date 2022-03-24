import numpy as np
import xarray as xr

from power_perceiver.data_loader.data_loader import BatchKey, DataLoader, NumpyBatch
from power_perceiver.utils import datetime64_to_int


class PV(DataLoader):
    @staticmethod
    def to_numpy(dataset: xr.Dataset) -> NumpyBatch:
        """This is called from Dataset.__getitem__.

        This processes this modality's xr.Dataset, to convert the xr.Dataset
        into a dictionary mapping BatchKeys to numpy arrays, as documented
        in the BatchKey class.
        """
        batch: NumpyBatch = {}

        # PV power
        # Note that, in v15 of the dataset, the keys are incorrectly named
        # power_mw and capacity_mwp, even though the power and capacity are both in watts.
        # See https://github.com/openclimatefix/nowcasting_dataset/issues/530
        # Also note that some capacities are 0. This will be fixed upstream in:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/622
        pv_normalised = dataset["power_mw"] / dataset["capacity_mwp"]
        batch[BatchKey.pv] = pv_normalised.values

        # In v15 of the dataset, `pv_system_row_number` is int64. This will be fixed in:
        # https://github.com/openclimatefix/nowcasting_dataset/issues/624
        batch[BatchKey.pv_system_row_number] = dataset["pv_system_row_number"].values.astype(
            np.int32
        )

        # Compute mask of valid PV data.
        # The mask will be a bool DataArray of shape (batch_size, n_pv_systems).
        valid_pv_capacity = dataset["capacity_mwp"] > 0
        # A NaN ID value is the "official" way to indicate a missing or deselected PV system.
        valid_pv_id = np.isfinite(dataset["id"])
        valid_pv_power = np.isfinite(pv_normalised).all(dim="time_index")
        pv_mask = valid_pv_capacity & valid_pv_id & valid_pv_power
        assert pv_mask.any(), "No valid PV systems!"
        batch[BatchKey.pv_mask] = pv_mask.values

        # Coordinates
        batch[BatchKey.pv_time_utc] = datetime64_to_int(dataset["time"].values)
        for batch_key, dataset_key in (
            (BatchKey.pv_x_osgb, "x_coords"),
            (BatchKey.pv_y_osgb, "y_coords"),
        ):
            coords = dataset[dataset_key]
            # PV spatial coords are float64 in v15. This will be fixed in:
            # https://github.com/openclimatefix/nowcasting_dataset/issues/624
            coords = coords.astype(np.float32)
            # Coords for missing PV systems are set to 0 in v15. This is dangerous! So set to NaN.
            # We need to set them to NaN so `np.nanmax()` does the right thing in `EncodeSpaceTime`
            # This won't be necessary after this issue is closed:
            # https://github.com/openclimatefix/nowcasting_dataset/issues/625
            coords = coords.where(pv_mask)
            batch[batch_key] = coords.values

        return batch
