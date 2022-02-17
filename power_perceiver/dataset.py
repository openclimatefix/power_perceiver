import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import fsspec
import numpy as np
import torch
import torch.utils.data
import xarray as xr

from power_perceiver.consts import BatchKey, DataSourceName
from power_perceiver.data_loader import DATA_SOURCE_NAME_TO_LOADER_CLASS, DataLoader

_log = logging.getLogger(__name__)


@dataclass
class NowcastingDataset(torch.utils.data.Dataset):
    """
    Attributes:
        data_path: Base path to the pre-prepared dataset.  e.g. .../v15/train/
        data_source_names: The names of the data sources. Must also be the name of the subdirectory
        max_n_batches_per_epoch: If the user sets this to an int then
            this int will be the max number of batches used per epoch. If left as None
            then will load as many batches as are available.
        n_batches: int. Set by _set_number_of_batches.
        data_source_loaders: dict[DataSourceName, DataLoader]
    """

    data_path: Path
    data_source_names: Iterable[DataSourceName]
    max_n_batches_per_epoch: Optional[int] = None

    def __post_init__(self):
        # Sanity checks
        assert self.data_path.exists()
        assert len(self.data_source_names) > 0
        self._instantiate_data_source_loaders()
        self._set_number_of_batches()

    def _instantiate_data_source_loaders(self) -> None:
        self.data_source_loaders: dict[DataSourceName, DataLoader] = {}
        for data_source_name in self.data_source_names:
            data_source_loader_class = DATA_SOURCE_NAME_TO_LOADER_CLASS[data_source_name]
            data_source_loader = data_source_loader_class(data_path=self.data_path)
            self.data_source_loaders[data_source_name] = data_source_loader

    def _set_number_of_batches(self) -> None:
        """Set number of batches.  Check every data source."""
        self.n_batches = None
        for data_source_name, data_source_loader in self.data_source_loaders.items():
            n_batches_for_data_source = data_source_loader.get_n_batches_available()
            if self.n_batches is None:
                self.n_batches = n_batches_for_data_source
            elif n_batches_for_data_source != self.n_batches:
                self.n_batches = min(self.n_batches, n_batches_for_data_source)
                _log.warning(
                    f"Warning! {data_source_name} has a different number of batches to at"
                    " least one other modality!"
                    f" We'll use the minimum of the two values: {self.n_batches}"
                )
        if self.max_n_batches_per_epoch is not None:
            self.n_batches = min(self.n_batches, self.max_n_batches_per_epoch)
        assert self.n_batches is not None
        assert self.n_batches > 0

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, batch_idx: int) -> torch.Tensor:
        np_data: dict[BatchKey, np.ndarray] = {}
        for data_source_name, data_source_loader in self.data_source_loaders.items():
            np_data_for_data_source = data_source_loader[batch_idx]
            np_data.update(np_data_for_data_source)
        # TODO: Convert to Tensors
        return np_data
