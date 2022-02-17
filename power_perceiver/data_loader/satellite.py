import numpy as np

from power_perceiver.data_loader import DataLoader


class SatelliteDataLoader(DataLoader):
    @property
    def data_source_name(self) -> str:
        return "satellite"

    def __getitem__(self, batch_idx: int) -> np.ndarray:
        xr_dataset = self.load_batch(batch_idx)
        # TODO: Convert this!
        return xr_dataset
