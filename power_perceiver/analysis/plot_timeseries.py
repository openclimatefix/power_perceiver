from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from power_perceiver.analysis.simple_callback import SimpleCallback
from power_perceiver.consts import BatchKey


def plot_pv_power(
    actual_pv_power: torch.Tensor,
    predicted_pv_power: torch.Tensor,
    example_idx: int,
    datetimes: torch.Tensor,
) -> plt.Figure:
    fig, (ax_actual, ax_predicted) = plt.subplots(nrows=2, sharex=True, sharey=True)

    datetimes = pd.to_datetime(datetimes[example_idx], unit="s")

    def _plot(ax, data, title):
        ax.plot(datetimes, data[example_idx].squeeze())
        ax.set_title(title)
        ax.set_ylabel("PV power")
        ax.set_xlabel("Timestep")

    _plot(ax_actual, actual_pv_power, "Actual PV power")
    _plot(ax_predicted, predicted_pv_power, "Predicted PV power")
    return fig


class LogTimeseriesPlots(SimpleCallback):
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
        if tag == "train":
            return
        EXAMPLE_IDX = 0
        predicted_pv_power = outputs["predicted_pv_power"].cpu().detach()
        actual_pv_power = batch[BatchKey.pv].cpu()[:, 12:30]
        datetimes = batch[BatchKey.pv_time_utc].cpu()[:, 12:30]
        if batch_idx < 4:
            fig = plot_pv_power(
                actual_pv_power=actual_pv_power,
                predicted_pv_power=predicted_pv_power,
                example_idx=EXAMPLE_IDX,
                datetimes=datetimes,
            )
            wandb.log(
                {
                    f"{tag}/pv_power/{batch_idx=}": wandb.Image(fig),
                    "global_step": trainer.global_step,
                },
            )
            plt.close(fig)
