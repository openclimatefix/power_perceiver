from typing import Any, Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb

from power_perceiver.analysis.simple_callback import SimpleCallback
from power_perceiver.consts import BatchKey
from power_perceiver.pytorch_modules.mixture_density_network import plot_probs


def plot_pv_power(
    actual_pv_power: torch.Tensor,  # shape: example, time, n_pv_systems
    predicted_pv_power: torch.Tensor,
    actual_gsp_power: torch.Tensor,
    predicted_gsp_power: torch.Tensor,
    example_idx: int,
    pv_datetimes: torch.Tensor,
    gsp_datetimes: torch.Tensor,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    pv_datetimes = pd.to_datetime(pv_datetimes[example_idx], unit="s")
    gsp_datetimes = pd.to_datetime(gsp_datetimes[example_idx], unit="s")

    # PV
    def _plot_pv(ax, data, title):
        ax.plot(pv_datetimes, data[example_idx].squeeze())
        ax.set_title(title)
        ax.set_ylabel("PV power")

    for pv_idx, ax in enumerate(axes[:-1]):
        title = "Actual PV power"
        if pv_idx == 1 and not pd.isnull(pv_datetimes[0]):
            title += "\nDate: " + pv_datetimes[0].date().strftime("%Y-%m-%d")
        _plot_pv(ax, actual_pv_power[example_idx, :, pv_idx], "Actual PV power")
        plot_probs(
            ax=ax,
            network_output=predicted_pv_power[example_idx, :, pv_idx],
            left=pv_datetimes[0],
            right=pv_datetimes[-1],
        )

    # GSP
    ax_gsp = axes[-1]
    ax_gsp.plot(gsp_datetimes, actual_gsp_power[example_idx], label="Actual")
    plot_probs(
        ax=ax_gsp,
        network_output=predicted_gsp_power[example_idx].squeeze(),
        left=gsp_datetimes[0],
        right=gsp_datetimes[-1],
    )
    ax_gsp.set_title("GSP PV power")
    ax_gsp.legend()

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    return fig


class LogProbabilityTimeseriesPlots(SimpleCallback):
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
            predicted_pv_power = outputs["predicted_pv_power"].cpu().detach()
            actual_pv_power = outputs["actual_pv_power"].cpu().detach()
            # TODO: Generate pv_datetimes upstream and pass it into this function, just like
            # we do with `gsp_time_utc`.
            pv_datetimes = batch[BatchKey.pv_time_utc].cpu()
            n_examples_per_batch = pv_datetimes.shape[0]
            examples_with_gsp_data = [
                example_idx
                for example_idx in range(n_examples_per_batch)
                if batch[BatchKey.gsp_time_utc][example_idx, 0].isfinite()
            ]
            for example_idx in examples_with_gsp_data[:4]:
                fig = plot_pv_power(
                    actual_pv_power=actual_pv_power,
                    predicted_pv_power=predicted_pv_power,
                    actual_gsp_power=outputs["actual_gsp_power"].cpu().detach(),
                    predicted_gsp_power=outputs["predicted_gsp_power"].cpu().detach(),
                    gsp_datetimes=outputs["gsp_time_utc"].cpu().detach(),
                    example_idx=example_idx,
                    pv_datetimes=pv_datetimes,
                )
                wandb.log(
                    {
                        f"{tag}/pv_power_probs/{batch_idx=};{example_idx=}": wandb.Image(fig),
                    },
                )
                plt.close(fig)
