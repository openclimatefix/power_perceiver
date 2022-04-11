import functools
from typing import Any, Optional

import pytorch_lightning as pl


class SimpleCallback(pl.Callback):
    """A simple abstract class which redirects `on_train_batch_end` and `on_validation_batch_end`
    to a single method, `_on_batch_end`."""

    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[dict[str, object]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
        tag: str,
    ) -> None:
        """Called when the batch ends.

        Args:
            outputs: The output from Model.training_step
            tag: 'train' or 'validation'
        """
        raise NotImplementedError  # MUST BE IMPLEMENTED BY SUBCLASSES!

    on_train_batch_end = functools.partialmethod(_on_batch_end, tag="train")
    on_validation_batch_end = functools.partialmethod(_on_batch_end, tag="validation")
