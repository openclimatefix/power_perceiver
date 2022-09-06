import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import torch
import torch.utils.data
from ocf_datapipes.utils.consts import BatchKey

from power_perceiver.load_prepared_batches.data_sources import NumpyBatch, PreparedDataSource
from power_perceiver.load_prepared_batches.data_sources.prepared_data_source import XarrayBatch

_log = logging.getLogger(__name__)


@dataclass
class PreparedDataset(torch.utils.data.Dataset):
    """Load batches pre-prepared by `nowcasting_dataset`.

    Initialisation arguments:
        data_loaders: A list of instantiated data loader objects.
        data_path: Base path to the pre-prepared dataset. e.g. /path/to/v15/train/
        max_n_batches_per_epoch: If the user sets this to an int then
            this int will be the max number of batches used per epoch. If left as None
            then will load as many batches as are available.
        xr_batch_processors: Functions which takes an XarrayBatch,
            and does processing *across* modalities, and returns the processed XarrayBatch.
            Note that and processing *within* a modality should be done in
            PreparedDataSource.to_numpy.
        np_batch_processors: Functions which takes a NumpyBatch,
            and does processing *across* modalities, and returns the processed NumpyBatch.
            Note that and processing *within* a modality should be done in
            PreparedDataSource.to_numpy.

    Attributes:
        n_batches: int. Set by _set_number_of_batches.
    """

    data_loaders: Iterable[PreparedDataSource]
    data_path: Optional[Path] = None
    max_n_batches_per_epoch: Optional[int] = None
    xr_batch_processors: Optional[Iterable[Callable]] = None
    np_batch_processors: Optional[Iterable[Callable]] = None

    def __post_init__(self):
        # Sanity checks
        assert self.data_path.exists()
        assert len(self.data_loaders) > 0
        # Prepare PreparedDataSources.
        self._set_data_path_in_data_loaders()
        self._set_number_of_batches()
        super().__init__()

    def _set_data_path_in_data_loaders(self) -> None:
        for data_loader in self.data_loaders:
            data_loader.data_path = self.data_path

    def _set_number_of_batches(self) -> None:
        """Set number of batches.  Check every data source."""
        self.n_batches = None
        for data_loader in self.data_loaders:
            n_batches_for_data_source = data_loader.get_n_batches_available()
            if self.n_batches is None:
                self.n_batches = n_batches_for_data_source
            elif n_batches_for_data_source != self.n_batches:
                self.n_batches = min(self.n_batches, n_batches_for_data_source)
                _log.warning(
                    f"Warning! {data_loader} has a different number of batches to at"
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
        if batch_idx >= self.n_batches:
            raise KeyError(f"{batch_idx=} is out of bounds! {self.n_batches=}")
        xr_batch = self._get_xarray_batch(batch_idx)
        xr_batch = self._process_xr_batch(xr_batch)
        np_batch = self._xarray_to_numpy_batch(xr_batch)
        del xr_batch
        np_batch = self._process_np_batch(np_batch)
        return np_batch

    def _get_xarray_batch(self, batch_idx: int) -> NumpyBatch:
        """Load the completely un-modified batches from disk and store them in a dict."""
        xr_batch: NumpyBatch = {}
        for data_loader in self.data_loaders:
            xr_data_for_data_source = data_loader[batch_idx]
            xr_batch[data_loader.__class__] = xr_data_for_data_source
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
        for data_loader_class, xr_dataset in xr_batch.items():
            if data_loader_class == BatchKey.requested_timesteps:
                # `ReduceNumTimesteps` introduces a `requested_timesteps` key,
                # whose value is a np.ndarray.
                requested_timesteps = xr_dataset
                np_batch[BatchKey.requested_timesteps] = requested_timesteps
            else:
                np_data_for_data_source = data_loader_class.to_numpy(xr_dataset)
                np_batch.update(np_data_for_data_source)
        return np_batch

    def _process_np_batch(self, np_batch: NumpyBatch) -> NumpyBatch:
        """If necessary, do any processing which needs to be done across modalities,
        on the NumpyBatch."""
        if self.np_batch_processors:
            for np_batch_processor in self.np_batch_processors:
                np_batch = np_batch_processor(np_batch)
        return np_batch
