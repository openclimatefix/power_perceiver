from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from power_perceiver.analysis.simple_callback import SimpleCallback
from power_perceiver.consts import NUM_HIST_SAT_IMAGES, BatchKey


def plot_satellite(
    actual_sat: torch.Tensor,  # shape: (example, time, y, x)
    predicted_sat: torch.Tensor,
    example_idx: torch.Tensor,
    sat_datetimes: torch.Tensor,
    num_timesteps: int = 4,
    interval: int = 6,
) -> plt.Figure:

    datetimes = pd.to_datetime(sat_datetimes[example_idx], unit="s")

    fig, axes = plt.subplots(nrows=2, ncols=num_timesteps)
    for ax, tensor, title in zip(axes, (actual_sat, predicted_sat), ("actual", "predicted")):
        for i in range(num_timesteps):
            timestep_idx = int(i * interval)
            dt = datetimes[timestep_idx + NUM_HIST_SAT_IMAGES]
            dt = dt.strftime("%Y-%m-%d %H:%M")
            if title == "actual":
                timestep_idx += NUM_HIST_SAT_IMAGES
            image = tensor[example_idx, timestep_idx]
            ax[i].imshow(image)
            ax[i].set_title(f"{title} {timestep_idx=} {dt}")

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
            sat_datetimes = batch[BatchKey.hrvsatellite_time_utc].cpu()
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
