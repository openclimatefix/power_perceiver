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
    predicted_pv_power_mean: torch.Tensor,
    predicted_gsp_power_mean: torch.Tensor,
    gsp_id: torch.Tensor,
    pv_id: torch.Tensor,
    nwp: torch.Tensor,
    nwp_target_time_utc: torch.Tensor,
    nwp_channel_names: torch.Tensor,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=3, ncols=4)
    axes = np.array(axes).flatten()

    pv_datetimes = pd.to_datetime(pv_datetimes[example_idx], unit="s")
    gsp_datetimes = pd.to_datetime(gsp_datetimes[example_idx], unit="s")
    nwp_datetimes = pd.to_datetime(nwp_target_time_utc[example_idx], unit="s")

    if pd.isnull(pv_datetimes[0]):
        # This example has no PV data.
        return fig

    legend_kwargs = dict(framealpha=0.4, fontsize="x-small")

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
            pv_datetimes, predicted_pv_power_mean[example_idx, :, pv_idx], label="Mean prediction"
        )
        ax.legend(**legend_kwargs)

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
    ax_gsp.legend(**legend_kwargs)

    # NWP
    ax_nwp = axes[-3]
    ax_nwp.set_title("NWP (mean over x and y)")
    ax_nwp_twin = ax_nwp.twiny()
    ax_nwp_twin.plot(
        nwp_datetimes,
        nwp[example_idx].mean(dim=(-1, -2)),
        label=nwp_channel_names,
    )
    ax_nwp_twin.legend(loc="upper right", **legend_kwargs)
    ax_nwp.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_nwp_twin.tick_params(
        axis="x", which="both", bottom=True, labelbottom=True, top=False, labeltop=False
    )

    # Format all the timeseries plots (PV and GSP and NWP)
    for ax in np.concatenate((axes[:-2], [ax_nwp_twin])):
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.tick_params(axis="x", labelsize="x-small")

    # Satellite
    sat_axes = [ax.twinx().twiny() for ax in axes[-2:]]

    actual_satellite = actual_satellite[example_idx, :, 0]
    sat_axes[0].imshow(actual_satellite[0])
    sat_axes[0].set_xlabel("First actual satellite image")

    sat_axes[1].imshow(actual_satellite[-1])
    sat_axes[1].set_xlabel("Last actual satellite image")

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

    for ax in axes[-2:]:
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
            pv_datetimes = outputs["pv_time_utc"].cpu().detach()
            n_examples_per_batch = pv_datetimes.shape[0]
            examples_with_gsp_data = [
                example_idx
                for example_idx in range(n_examples_per_batch)
                if batch[BatchKey.gsp_time_utc][example_idx].isfinite().all()
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
                pv_id=batch[BatchKey.pv_id].cpu(),
                gsp_id=batch[BatchKey.gsp_id].squeeze().cpu(),
                nwp=batch[BatchKey.nwp].cpu(),
                nwp_target_time_utc=batch[BatchKey.nwp_target_time_utc].cpu(),
                nwp_channel_names=batch[BatchKey.nwp_channel_names],
            )
            pl_module.logger.experiment.log(
                {f"{tag}/pv_power_probs/{batch_idx=};{example_idx=}": wandb.Image(fig)}
            )
            plt.close(fig)
