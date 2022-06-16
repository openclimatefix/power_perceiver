from typing import Any, Optional

import matplotlib.dates as mdates
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
    actual_gsp_power: torch.Tensor,
    predicted_gsp_power: torch.Tensor,
    example_idx: int,
    pv_datetimes: torch.Tensor,
    gsp_datetimes: torch.Tensor,
    solar_azimuth: torch.Tensor,
    solar_elevation: torch.Tensor,
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=4, sharex=True, sharey=True)
    ax_pv_actual, ax_pv_predicted, ax_gsp, ax_solar = axes

    pv_datetimes = pd.to_datetime(pv_datetimes[example_idx], unit="s")
    gsp_datetimes = pd.to_datetime(gsp_datetimes[example_idx], unit="s")

    # PV
    def _plot_pv(ax, data, title):
        ax.plot(pv_datetimes, data[example_idx].squeeze())
        ax.set_title(title)
        ax.set_ylabel("PV power")

    _plot_pv(ax_pv_actual, actual_pv_power, "Actual PV power")
    _plot_pv(ax_pv_predicted, predicted_pv_power, "Predicted PV power")

    # GSP
    gsp_df = pd.DataFrame(
        {
            "actual_gsp_power": actual_gsp_power[example_idx].squeeze(),
            "predicted_gsp_power": predicted_gsp_power[example_idx].squeeze(),
        },
        index=gsp_datetimes,
    )
    gsp_index_dupes = gsp_df.index.duplicated()
    ax_gsp.plot(
        gsp_df.index[~gsp_index_dupes],
        gsp_df["actual_gsp_power"].loc[~gsp_index_dupes],
        label="Actual",
    )
    ax_gsp.scatter(gsp_df.index, gsp_df["predicted_gsp_power"], alpha=0.8, label="Predicted")
    ax_gsp.set_title("GSP PV power")
    ax_gsp.set_ylabel("GSP PV power")
    ax_gsp.legend()

    # Solar elevation and azimuth
    ax2_solar = ax_solar.twinx()  # Don't share Y axis!
    ax2_solar.plot(pv_datetimes, solar_azimuth[example_idx], label="Solar Azimuth")
    ax2_solar.plot(pv_datetimes, solar_elevation[example_idx], label="Solar Elevation")
    ax2_solar.legend()
    ax2_solar.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    xlabel = "Time (UTC)"
    if not pd.isnull(pv_datetimes[0]):  # Check this example is not NaT!
        xlabel += "\nDate: " + pv_datetimes[0].date().strftime("%Y-%m-%d")
    ax2_solar.set_xlabel(xlabel)

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
        if batch_idx < 4:
            predicted_pv_power = outputs["predicted_pv_power_mean"].cpu().detach()
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
                    predicted_gsp_power=outputs["predicted_gsp_power_mean"].cpu().detach(),
                    gsp_datetimes=outputs["gsp_time_utc"].cpu().detach(),
                    example_idx=example_idx,
                    pv_datetimes=pv_datetimes,
                    solar_azimuth=batch[BatchKey.hrvsatellite_solar_azimuth].cpu().detach(),
                    solar_elevation=batch[BatchKey.hrvsatellite_solar_elevation].cpu().detach(),
                )
                pl_module.logger.experiment.log(
                    {
                        f"{tag}/pv_power/{batch_idx=};{example_idx=}": wandb.Image(fig),
                        "global_step": trainer.global_step,
                    },
                )
                plt.close(fig)
