import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.utils.data

from power_perceiver.consts import DataSourceName, NumpyBatch, XarrayBatch
from power_perceiver.data_loader import DATA_SOURCE_NAME_TO_LOADER_CLASS, DataLoader

_log = logging.getLogger(__name__)


@dataclass
class NowcastingDataset(torch.utils.data.Dataset):
    """
    Initialisation arguments:
        data_path: Base path to the pre-prepared dataset. e.g. /path/to/v15/train/
        data_source_names: The names of the requested data sources. Must also be the name of
            the subdirectory in which the data resides.
        max_n_batches_per_epoch: If the user sets this to an int then
            this int will be the max number of batches used per epoch. If left as None
            then will load as many batches as are available.
        xr_batch_processors: Functions which takes a dict[DataSourceName, xr.Dataset],
            and does processing *across* modalities, and returns a dict[DataSourceName, xr.Dataset].
            Note that and processing *within* a modality should be done in
            DataLoader.to_numpy.

    Attributes:
        data_source_loaders: dict[DataSourceName, DataLoader]
        n_batches: int. Set by _set_number_of_batches.
    """

    data_path: Path
    data_source_names: Iterable[DataSourceName]
    max_n_batches_per_epoch: Optional[int] = None
    xr_batch_processors: Optional[Iterable[Callable]] = None

    def __post_init__(self):
        # Sanity checks
        assert self.data_path.exists()
        assert len(self.data_source_names) > 0
        # Prepare DataLoaders.
        self._instantiate_data_source_loaders()
        self._set_number_of_batches()

    def _instantiate_data_source_loaders(self) -> None:
        self.data_source_loaders: dict[DataSourceName, DataLoader] = {}
        for data_source_name in self.data_source_names:
            data_source_loader_class = DATA_SOURCE_NAME_TO_LOADER_CLASS[data_source_name]
            data_source_loader = data_source_loader_class(
                data_path=self.data_path, data_source_name=data_source_name
            )
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

    def __getitem__(self, batch_idx: int) -> NumpyBatch:
        xr_batch = self._get_xarray_batch(batch_idx)
        xr_batch = self._process_xr_batch(xr_batch)
        np_batch = self._xarray_to_numpy_batch(xr_batch)
        return np_batch

    def _get_xarray_batch(self, batch_idx: int) -> XarrayBatch:
        """Load the completely un-modified batches from disk and store them in a dict."""
        xr_batch: XarrayBatch = {}
        for data_source_name, data_source_loader in self.data_source_loaders.items():
            xr_data_for_data_source = data_source_loader[batch_idx]
            xr_batch[data_source_name] = xr_data_for_data_source
        return xr_batch

    def _process_xr_batch(self, xr_batch: XarrayBatch) -> XarrayBatch:
        """If necessary, do any processing which needs to be done across modalities,
        on the xr.Datasets."""
        if self.xr_batch_processors:
            for xr_batch_processor in self.xr_batch_processors:
                xr_batch = xr_batch_processor(xr_batch)
        return xr_batch

    def _xarray_to_numpy_batch(self, xr_batch: XarrayBatch) -> NumpyBatch:
        """Convert from xarray Datasets to numpy."""
        np_batch: NumpyBatch = {}
        for data_source_name, xr_dataset in xr_batch.items():
            data_source_obj = self.data_source_loaders[data_source_name]
            np_data_for_data_source = data_source_obj.to_numpy(xr_dataset)
            np_batch.update(np_data_for_data_source)
        return np_batch
