import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import fsspec
import numpy as np
import xarray as xr
from ocf_datapipes.utils.consts import BatchKey

_log = logging.getLogger(__name__)


NumpyBatch = dict[BatchKey, np.ndarray]


@dataclass
class PreparedDataSource:
    """Load each data source.

    Initialisation arguments:
        data_path (Path): Optional.
        filename_suffix (str): Without the period (.)
        transforms: A list of transform functions. Each must accept an xr.Dataset, and must
            return a transformed xr.Dataset.

    How to add a new subclass:
      1. Create new subclass :)
      2. Override / set:
         - PreparedDataSource.to_numpy
      3. If necessary, also update BatchKey in consts.py
    """

    data_path: Optional[Path] = None
    filename_suffix: str = "nc"
    transforms: Optional[Iterable[Callable]] = None

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__.lower()

    @property
    def full_data_path(self) -> Path:
        try:
            return self.data_path / self.name
        except:  # noqa: E722
            if self.data_path is None:
                raise ValueError("data_path must be set!")
            else:
                raise

    def __getitem__(self, batch_idx: int) -> xr.Dataset:
        filename = self.get_filename(batch_idx=batch_idx)
        dataset = load_netcdf(filename)
        dataset = self.process_before_transforms(dataset)
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

    def process_before_transforms(self, dataset: xr.Dataset) -> xr.Dataset:
        """Can be overridden by subclass."""
        return dataset

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


# The one exception to this type, is that `ReduceNumTimesteps`
# adds a `BatchKey.requested_timesteps` key, whose value is a np.ndarray.
XarrayBatch = dict[PreparedDataSource.__class__, xr.Dataset]
