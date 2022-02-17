import logging
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

_log = logging.getLogger(__name__)


@dataclass
class DataLoader:
    data_path: Path

    def get_filename(self, batch_idx: int) -> Path:
        return self.data_path / self.data_source_name / f"{batch_idx:06d}.{self.suffix}"

    @property
    def suffix(self) -> str:
        return ".nc"

    def load_batch(self, batch_idx: int) -> xr.Dataset:
        filename = self.get_filename(batch_idx=batch_idx)
        return load_netcdf(filename)

    def get_n_batches_available(self) -> int:
        path_for_data_source = self.data_path / self.data_source_name
        n_batches = len(list(path_for_data_source.glob(f"*{self.suffix}")))
        _log.info(f"{self.data_source_name} has {n_batches} batches.")
        return n_batches

    # METHODS THAT MUST BE OVERRIDDEN ##################################
    # Override the self.data_source_name property and the methods below:
    def __getitem__(self, batch_idx: int) -> np.ndarray:
        raise NotImplementedError()


def load_netcdf(filename, engine="h5netcdf", *args, **kwargs) -> xr.Dataset:
    """Load a NetCDF dataset from local file system or cloud bucket."""
    with fsspec.open(filename, mode="rb") as file:
        dataset = xr.load_dataset(file, engine=engine, *args, **kwargs)
    return dataset
