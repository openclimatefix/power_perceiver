import logging
from dataclasses import dataclass
from numbers import Number
from typing import Callable, Iterable, Optional

import numpy as np
import torch

from power_perceiver.load_raw.data_loader.raw_data_loader import RawDataLoader

_log = logging.getLogger(__name__)


@dataclass
class RawDataset(torch.utils.data.Dataset):
    """
    Initialisation arguments:
        data_loaders: A dict where the keys are strings (short, arbitrary names
            of the example type), and the values are tuples or lists of
            instantiated data loader objects. The first data loader
            will be used to randomly select the location of each example.
            e.g. `dict(hrv_only=(HRV(),), hrv_and_pv=(HRV(), PV()))`
        probability_of_each_data_loader: A dict where the key is the same
            short identifier of each example type, and the key is the
            probability of loading that example type. Probabilities must
            sum to 1.
        xr_batch_processors: Functions which takes an XarrayBatch,
            and does processing *across* modalities, and returns the processed XarrayBatch.
            Note that and processing *within* a modality should be done in
            DataLoader.to_numpy.
        np_batch_processors: Functions which takes a NumpyBatch,
            and does processing *across* modalities, and returns the processed NumpyBatch.
            Note that and processing *within* a modality should be done in
            DataLoader.to_numpy.
    """

    data_loaders: dict[str, Iterable[RawDataLoader]]
    probability_of_each_data_loader: dict[str, Number]
    xr_batch_processors: Optional[Iterable[Callable]] = None
    np_batch_processors: Optional[Iterable[Callable]] = None

    def __post_init__(self):
        super().__init__()

    def per_worker_init(self, worker_id: int = 0) -> None:
        """Called by worker_init_fn on each copy of this dataset after the
        worker process has been spawned."""
        self.worker_id = worker_id
        # Each worker must have a different seed for its random number generator.
        # Otherwise all the workers will output exactly the same data!
        seed = torch.initial_seed()
        _log.info(f"{worker_id=} has random number generator {seed=:,d}")
        self.rng = np.random.default_rng(seed=seed)

        for data_loader in self._unique_data_loaders:
            data_loader.per_worker_init(worker_id=worker_id)

    @property
    def _unique_data_loaders(self):
        data_loaders = []
        for tuple_of_data_loaders in self.data_loaders.values():
            for data_loader in tuple_of_data_loaders:
                data_loaders.append(data_loader)
        return np.unique(data_loaders)
