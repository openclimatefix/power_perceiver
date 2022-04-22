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
) -> plt.Figure:
    fig, axes = plt.subplots(nrows=3, sharex=True, sharey=True)
    ax_pv_actual, ax_pv_predicted, ax_gsp = axes

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
    ax_gsp.set_xlabel("Time (UTC)\nDate: " + pv_datetimes[0].date().strftime("%Y-%m-%d"))
    ax_gsp.legend()

    ax_gsp.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    return fig


class LogTimeseriesPlots(SimpleCallback):
    def _on_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[dict[str, object]],
        batch: Any,
        batch_idx: int,
        tag: str,
    ) -> None:
        """Called when the training batch ends.

        Args:
            outputs: The output from Model.training_step
            tag: train or validation
        """
        if tag == "train":
            # We currently only train on a single timestep at a time, so not much point
            # plotting a timeseries!
            return

        if tag == "validation" and batch_idx < 4:
            predicted_pv_power = outputs["predicted_pv_power"].cpu().detach()
            actual_pv_power = batch[BatchKey.pv].cpu()[:, 6:-3]
            pv_datetimes = batch[BatchKey.pv_time_utc].cpu()[:, 6:-3]
            for example_idx in range(4):
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
                        f"{tag}/pv_power/{batch_idx=};{example_idx=}": wandb.Image(fig),
                        "global_step": trainer.global_step,
                    },
                )
                plt.close(fig)
