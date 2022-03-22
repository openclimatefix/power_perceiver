import numpy as np
import xarray as xr

from power_perceiver.data_loader.data_loader import BatchKey, DataLoader, NumpyBatch


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
        # power_mw and capacity_mwp, even though the power are capacity are both in watts.
        # See https://github.com/openclimatefix/nowcasting_dataset/issues/530
        pv_normalised = dataset["power_mw"] / dataset["capacity_mwp"]
        batch[BatchKey.pv] = pv_normalised.values

        batch[BatchKey.pv_system_row_number] = dataset["pv_system_row_number"].values.astype(
            np.int32
        )

        # Compute mask of valid PV data
        valid_pv_capacity = dataset["capacity_mwp"] > 0
        valid_pv_id = np.isfinite(dataset["id"])
        valid_pv_power = np.isfinite(pv_normalised).all(dim="time_index")
        pv_mask = valid_pv_capacity & valid_pv_id & valid_pv_power
        assert pv_mask.any(), "No valid PV systems!"
        batch[BatchKey.pv_mask] = pv_mask.values

        return batch
