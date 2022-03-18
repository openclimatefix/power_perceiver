import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import fsspec
import numpy as np
import xarray as xr

from power_perceiver.consts import BatchKey

_log = logging.getLogger(__name__)


NumpyBatch = dict[BatchKey, np.ndarray]


@dataclass
class DataLoader:
    """Load each data source.

    Initialisation arguments:
        filename_suffix (str): Without the period (.)
        transforms: A list of transform functions. Each must accept an xr.Dataset, and must
            return a transformed xr.Dataset.

    Attributes:
        data_path (Path): Actually, under the hood this is stored in a _data_class attribute,
            and set and got through setter and getter methods.
        full_data_path (Path): Set by the `data_path` setter.

    How to add a new subclass:
      1. Create new subclass :)
      2. Override / set:
         - DataLoader.to_numpy
      3. If necessary, also update BatchKey in consts.py
    """

    filename_suffix: str = "nc"
    transforms: Optional[Iterable[Callable]] = None

    def __post_init__(self) -> None:
        self._data_path = None

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__.lower()

    @property
    def data_path(self) -> Path:
        if self._data_path is None:
            raise ValueError("data_path is not set!")
        return self._data_path

    @data_path.setter
    def data_path(self, data_path: Path) -> None:
        self._data_path = data_path
        self.full_data_path = data_path / self.name

    def __getitem__(self, batch_idx: int) -> xr.Dataset:
        filename = self.get_filename(batch_idx=batch_idx)
        dataset = load_netcdf(filename)
        if self.transforms:
            for transform in self.transforms:
                dataset = transform(dataset)
        return dataset

    def get_filename(self, batch_idx: int) -> Path:
        return self.full_data_path / f"{batch_idx:06d}.{self.filename_suffix}"

    def get_n_batches_available(self) -> int:
        n_batches = len(list(self.full_data_path.glob(f"*.{self.filename_suffix}")))
        _log.info(f"{self.name} has {n_batches} batches.")
        return n_batches

    @staticmethod
    def to_numpy(dataset: xr.Dataset) -> NumpyBatch:
        """This is called from Dataset._xarray_to_numpy_batch.

        Process this modality's xr.Dataset, to convert the xr.Dataset
        into a dictionary mapping BatchKeys to numpy arrays, as documented
        in the BatchKey class.
        """
        raise NotImplementedError


def load_netcdf(filename, engine="h5netcdf", *args, **kwargs) -> xr.Dataset:
    """Load a NetCDF dataset from local file system or cloud bucket."""
    with fsspec.open(filename, mode="rb") as file:
        dataset = xr.load_dataset(file, engine=engine, *args, **kwargs)
    return dataset


XarrayBatch = dict[DataLoader.__class__, xr.Dataset]
