from typing import Any, Optional

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import wandb

from power_perceiver.analysis.simple_callback import SimpleCallback
from power_perceiver.consts import BatchKey


def plot_satellite(
    actual_sat: torch.Tensor,
    predicted_sat: torch.Tensor,
    example_idx: torch.Tensor,
    sat_datetimes: torch.Tensor,
) -> plt.Figure:
    fig, (axes_pred, axes_actual) = plt.subplots(nrows=2, ncols=4)

    print(f"{len(axes_pred)=}")
    print(f"{actual_sat.shape=}")
    print(f"{predicted_sat.shape=}")

    return fig


class LogSatellitePlots(SimpleCallback):
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
        """Called when the training batch ends.

        Args:
            outputs: The output from Model.training_step
            tag: train or validation
        """
        if batch_idx < 4:
            predicted_sat = outputs["predicted_sat"].cpu().detach()
            actual_sat = outputs["actual_sat"].cpu().detach()
            sat_datetimes = batch[BatchKey.sat_time_utc].cpu()
            for example_idx in range(4):
                fig = plot_satellite(
                    actual_sat=actual_sat,
                    predicted_sat=predicted_sat,
                    example_idx=example_idx,
                    sat_datetimes=sat_datetimes,
                )
                wandb.log(
                    {
                        f"{tag}/satellite/{batch_idx=};{example_idx=}": wandb.Image(fig),
                        "global_step": trainer.global_step,
                    },
                )
                plt.close(fig)
