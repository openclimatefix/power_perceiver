import logging
from typing import Any, Optional

import matplotlib
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

_log = logging.getLogger(__name__)


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
    predicted_pv_power_mean: torch.Tensor,
    predicted_gsp_power_mean: torch.Tensor,
    gsp_id: torch.Tensor,
    pv_id: torch.Tensor,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=3, ncols=4)
    axes = np.array(axes).flatten()

    pv_datetimes = pd.to_datetime(pv_datetimes[example_idx], unit="s")
    gsp_datetimes = pd.to_datetime(gsp_datetimes[example_idx], unit="s")

    if pd.isnull(pv_datetimes[0]):
        # This example has no PV data.
        return fig

    # PV
    for pv_idx, ax in enumerate(axes[:-4]):
        plot_probs(
            ax=ax,
            network_output=predicted_pv_power[example_idx, :, pv_idx],
            left=pv_datetimes[0],
            right=pv_datetimes[-1],
        )
        ax.set_title("PV power for {:.0f}".format(pv_id[example_idx, pv_idx]))
        try:
            ax.plot(pv_datetimes, actual_pv_power[example_idx, :, pv_idx], label="Actual PV")
        except Exception as e:
            raise e.__class__(
                f"{pv_datetimes.shape=}; {gsp_datetimes.shape=};"
                f" {actual_pv_power.shape=}; {predicted_pv_power.shape=}"
            ) from e
        ax.plot(
            pv_datetimes,
            pv_power_from_sat_transformer[example_idx, :, pv_idx],
            label="SatTrans prediction",
        )
        ax.plot(
            pv_datetimes, predicted_pv_power_mean[example_idx, :, pv_idx], label="Mean prediction"
        )
        ax.legend(framealpha=0.4)

    # GSP
    ax_gsp = axes[-4]
    plot_probs(
        ax=ax_gsp,
        network_output=predicted_gsp_power[example_idx].squeeze(),
        left=gsp_datetimes[0],
        right=gsp_datetimes[-1],
    )
    ax_gsp.plot(gsp_datetimes, actual_gsp_power[example_idx], label="Actual")
    ax_gsp.plot(gsp_datetimes, predicted_gsp_power_mean[example_idx], label="Mean prediction")
    ax_gsp.set_title("GSP PV power for {:.0f}".format(gsp_id[example_idx]))
    ax_gsp.set_xlabel(pv_datetimes[0].date().strftime("%Y-%m-%d"))
    ax_gsp.legend(framealpha=0.4)

    # Format all the timeseries plots (PV and GSP)
    for ax in axes[:-3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelsize="small")

    # Satellite
    sat_axes = [ax.twinx().twiny() for ax in axes[-3:]]

    actual_satellite = actual_satellite[example_idx, :, 0]
    sat_axes[0].imshow(actual_satellite[0])
    sat_axes[0].set_xlabel("First actual satellite image")

    sat_axes[1].imshow(actual_satellite[-1])
    sat_axes[1].set_xlabel("Last actual satellite image")

    sat_axes[2].imshow(surface_height[example_idx])
    sat_axes[2].set_xlabel("Surface height")

    def _turn_off_ticks(ax: matplotlib.axes.Axes):
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labelleft=False,
            labeltop=False,
            labelright=False,
        )

    for ax in axes[-3:]:
        _turn_off_ticks(ax)

    for ax in sat_axes:
        _turn_off_ticks(ax)

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
        if batch_idx in range(0, 128, 16):
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
            example_idx = examples_with_gsp_data[0]
            fig = plot_pv_power(
                actual_pv_power=actual_pv_power,
                predicted_pv_power=predicted_pv_power,
                predicted_pv_power_mean=outputs["predicted_pv_power_mean"].cpu().detach(),
                actual_gsp_power=outputs["actual_gsp_power"].cpu().detach(),
                predicted_gsp_power=outputs["predicted_gsp_power"].cpu().detach(),
                predicted_gsp_power_mean=outputs["predicted_gsp_power_mean"].cpu().detach(),
                gsp_datetimes=outputs["gsp_time_utc"].cpu().detach(),
                example_idx=example_idx,
                pv_datetimes=pv_datetimes,
                actual_satellite=batch[BatchKey.hrvsatellite_actual].cpu(),
                surface_height=batch[BatchKey.hrvsatellite_surface_height].cpu(),
                pv_power_from_sat_transformer=outputs["pv_power_from_sat_transformer"]
                .cpu()
                .detach(),
                pv_id=batch[BatchKey.pv_id].cpu(),
                gsp_id=batch[BatchKey.gsp_id].squeeze().cpu(),
            )
            pl_module.logger.experiment.log(
                {f"{tag}/pv_power_probs/{batch_idx=};{example_idx=}": wandb.Image(fig)}
            )
            plt.close(fig)
