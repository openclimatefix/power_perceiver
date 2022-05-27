from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from power_perceiver.analysis.simple_callback import SimpleCallback
from power_perceiver.consts import BatchKey


def plot_satellite(
    actual_sat: torch.Tensor,  # shape: (example, time, y, x)
    predicted_sat: torch.Tensor,
    example_idx: torch.Tensor,
    sat_datetimes: torch.Tensor,
    num_timesteps: int = 8,
    interval: int = 3,
) -> plt.Figure:

    datetimes = pd.to_datetime(sat_datetimes[example_idx], unit="s")
    actual_sat = actual_sat[example_idx]
    predicted_sat = predicted_sat[example_idx]

    # Use the same "exposure" for all plots:
    vmin = min(actual_sat.min(), predicted_sat.min())
    vmax = max(actual_sat.max(), predicted_sat.max())

    fig, axes = plt.subplots(nrows=2, ncols=num_timesteps)
    for ax, tensor, title in zip(axes, (actual_sat, predicted_sat), ("actual", "predicted")):
        for i in range(num_timesteps):
            timestep_idx = int(i * interval)
            dt = datetimes[timestep_idx]
            dt = dt.strftime("%Y-%m-%d %H:%M")
            image = tensor[timestep_idx]
            ax[i].imshow(image, vmin=vmin, vmax=vmax)
            ax[i].set_title(f"{title} {timestep_idx=}\n{dt}")

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
                pl_module.logger.experiment.log(
                    {
                        f"{tag}/satellite/{batch_idx=};{example_idx=}": wandb.Image(fig),
                    },
                )
                plt.close(fig)

                # Plot surface height:
                fig, ax = plt.subplots()
                ax.imshow(batch[BatchKey.hrvsatellite_surface_height][example_idx].cpu())
                pl_module.logger.experiment.log(
                    {f"{tag}/surface_height/{batch_idx=};{example_idx=}": wandb.Image(fig)}
                )
                plt.close(fig)
