import logging
from dataclasses import dataclass
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

from power_perceiver.consts import BatchKey, DataSourceName

_log = logging.getLogger(__name__)


@dataclass
class DataLoader:
    """Load each data source.

    Initialisation arguments:
        data_path:
        data_source_name:
        filename_suffix (str): Without the period (.)

    Attributes:
        full_data_path (Path): Set by __post_init__.

    How to add a new subclass:
      1. Create new subclass :). Override DataLoader.to_numpy
      2. Update DATA_SOURCE_NAME_TO_LOADER_CLASS in __init__.py
      3. If necessary, also update DataSourceName and/or BatchKey in consts.py
    """

    data_path: Path
    data_source_name: DataSourceName
    filename_suffix: str = "nc"

    def __post_init__(self) -> None:
        self.full_data_path = self.data_path / self.data_source_name.value

    def __getitem__(self, batch_idx: int) -> xr.Dataset:
        filename = self.get_filename(batch_idx=batch_idx)
        return load_netcdf(filename)

    def get_filename(self, batch_idx: int) -> Path:
        return self.full_data_path / f"{batch_idx:06d}.{self.filename_suffix}"

    def get_n_batches_available(self) -> int:
        n_batches = len(list(self.full_data_path.glob(f"*.{self.filename_suffix}")))
        _log.info(f"{self.data_source_name} has {n_batches} batches.")
        return n_batches

    def to_numpy(self, dataset: xr.Dataset) -> dict[BatchKey, np.ndarray]:
        """This is called from Dataset.__getitem__.

        This processes this modality's xr.Dataset, to convert the xr.Dataset
        into a dictionary mapping BatchKeys to numpy arrays, as documented
        in the BatchKey class.
        """
        raise NotImplementedError


def load_netcdf(filename, engine="h5netcdf", *args, **kwargs) -> xr.Dataset:
    """Load a NetCDF dataset from local file system or cloud bucket."""
    with fsspec.open(filename, mode="rb") as file:
        dataset = xr.load_dataset(file, engine=engine, *args, **kwargs)
    return dataset
