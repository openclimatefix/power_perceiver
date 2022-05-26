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
    actual_satellite: torch.Tensor,  # shape: example, time, channels, y, x
    surface_height: torch.Tensor,
    pv_power_from_sat_transformer: torch.Tensor,
    gsp_power_from_sat_transformer: torch.Tensor,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    pv_datetimes = pd.to_datetime(pv_datetimes[example_idx], unit="s")
    gsp_datetimes = pd.to_datetime(gsp_datetimes[example_idx], unit="s")

    if pd.isnull(pv_datetimes[0]):
        # This example has no PV data.
        return fig

    # PV
    def _plot_pv(ax, data, title):
        ax.plot(pv_datetimes, data)
        ax.set_title(title)
        ax.set_ylabel("PV power")

    for pv_idx, ax in enumerate(axes[:-5]):
        plot_probs(
            ax=ax,
            network_output=predicted_pv_power[example_idx, :, pv_idx],
            left=pv_datetimes[0],
            right=pv_datetimes[-1],
        )
        title = "Actual PV power"
        if pv_idx == 0:
            title += " " + pv_datetimes[0].date().strftime("%Y-%m-%d")
        _plot_pv(ax, actual_pv_power[example_idx, :, pv_idx], title)
        _plot_pv(ax, pv_power_from_sat_transformer[example_idx, :, pv_idx], "SatTrans prediction")
        ax.legend()

    # GSP
    ax_gsp = axes[-4]
    plot_probs(
        ax=ax_gsp,
        network_output=predicted_gsp_power[example_idx].squeeze(),
        left=gsp_datetimes[0],
        right=gsp_datetimes[-1],
    )
    ax_gsp.plot(gsp_datetimes, actual_gsp_power[example_idx], label="Actual")
    ax_gsp.plot(
        gsp_datetimes, gsp_power_from_sat_transformer[example_idx], label="SatTrans prediction"
    )
    ax_gsp.set_title("GSP PV power")
    ax_gsp.legend()

    for ax in axes[:-3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    # Satellite
    actual_satellite = actual_satellite[example_idx, :, 0]
    axes[-3].twinx().twiny().imshow(actual_satellite[0])
    axes[-3].set_title("First actual satellite image")
    axes[-2].twinx().twiny().imshow(actual_satellite[-1])
    axes[-2].set_title("Last actual satellite image")
    axes[-1].twinx().twiny().imshow(surface_height[example_idx])
    axes[-1].set_title("Surface height")
    for ax in axes[-3:]:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
        )

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
                if batch[BatchKey.pv_time_utc][example_idx].isfinite().all()
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
                    actual_satellite=batch[BatchKey.hrvsatellite].cpu(),
                    surface_height=batch[BatchKey.hrvsatellite_surface_height].cpu(),
                    pv_power_from_sat_transformer=outputs["pv_power_from_sat_transformer"]
                    .cpu()
                    .detach(),
                    gsp_power_from_sat_transformer=outputs["gsp_power_from_sat_transformer"]
                    .cpu()
                    .detach(),
                )
                wandb.log(
                    {
                        f"{tag}/pv_power_probs/{batch_idx=};{example_idx=}": wandb.Image(fig),
                    },
                )
                plt.close(fig)
