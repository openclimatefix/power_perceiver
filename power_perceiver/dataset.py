import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch.utils.data

from power_perceiver.consts import DataSourceName

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
        n_batches_available: int. Set by _set_number_of_batches.
    """

    data_path: Path
    data_source_names: Iterable[DataSourceName]
    max_n_batches_per_epoch: Optional[int] = None

    def __post_init__(self):
        # Sanity checks
        assert self.data_path.exists()
        assert len(self.data_source_names) > 0
        self._set_number_of_batches()

    def _set_number_of_batches(self) -> None:
        """Set number of batches.  Check every data source."""
        self.n_batches_available = None
        for data_source_name in self.data_source_names:
            path_for_data_source = self.data_path / data_source_name.value
            n_batches_for_data_source = len(list(path_for_data_source.glob("*.nc")))
            _log.info(f"{data_source_name} has {n_batches_for_data_source} batches.")
            if self.n_batches_available is None:
                self.n_batches_available = n_batches_for_data_source
            elif n_batches_for_data_source != self.n_batches_available:
                self.n_batches_available = min(self.n_batches_available, n_batches_for_data_source)
                _log.warning(
                    f"Warning! {data_source_name} has a different number of batches to at"
                    " least one other modality!"
                    f" We'll use the minimum of the two values: {self.n_batches_available}"
                )
        if self.max_n_batches_per_epoch is not None:
            self.n_batches_available = min(self.n_batches_available, self.max_n_batches_per_epoch)
        assert self.n_batches_available is not None
        assert self.n_batches_available > 0
