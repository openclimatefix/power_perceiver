import xarray as xr

from power_perceiver.consts import BatchKey, NumpyBatch
from power_perceiver.data_loader.data_loader import DataLoader


class PVLoader(DataLoader):
    def to_numpy(self, dataset: xr.Dataset) -> NumpyBatch:
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
        pv = dataset["power_mw"] / dataset["capacity_mwp"]
        pv = pv.values
        batch[BatchKey.pv] = pv
        del pv

        batch[BatchKey.pv_system_row_number] = dataset["pv_system_row_number"].values
        return batch
