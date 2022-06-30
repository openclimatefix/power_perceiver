from typing import Any, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import wandb
from matplotlib import pyplot as plt

from power_perceiver.consts import BatchKey


class LogNationalPV(pl.Callback):
    """To be used in conjunction with the `NationalPVDataset`. Must be used with a single
    validation data loader worker!

    Attributes:
        _national_pv_accumulator_mw: A pd.DataFrame with columns "actual" and "predicted"
            and index is the target DatetimeIndex (in UTC) of the data.
        _national_batch_idx: The number of complete national "cycles" completed.
    """

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._national_pv_accumulator_mw = None
        self._national_batch_idx = 0

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Optional[dict[str, torch.Tensor]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        predicted_gsp_power = outputs["predicted_gsp_power_mean"].cpu().detach()  # (example, time)
        actual_gsp_power = batch[BatchKey.gsp].squeeze().cpu()  # shape: (example, time)
        gsp_datetimes = batch[BatchKey.gsp_time_utc].cpu()  # shape: (example, time)
        gsp_capacity_mwp = batch[BatchKey.gsp_capacity_mwp].squeeze().cpu().numpy()  # (example,)
        num_examples = predicted_gsp_power.shape[0]

        gsp_t0_idx = batch[BatchKey.gsp_t0_idx]

        # Select just the forecast timesteps:
        predicted_gsp_power = predicted_gsp_power[:, gsp_t0_idx + 1 :]
        actual_gsp_power = actual_gsp_power[:, gsp_t0_idx + 1 :]
        gsp_datetimes = gsp_datetimes[:, gsp_t0_idx + 1 :]

        for example_idx in range(num_examples):
            dt_for_example = pd.to_datetime(gsp_datetimes[example_idx], unit="s")
            df_for_example = pd.DataFrame(
                {
                    "actual": actual_gsp_power[example_idx],
                    "predicted": predicted_gsp_power[example_idx],
                },
                index=dt_for_example,
            )
            df_for_example *= gsp_capacity_mwp[example_idx]

            if self._national_pv_accumulator_mw is None:
                self._national_pv_accumulator_mw = df_for_example
            elif np.array_equal(dt_for_example, self._national_pv_accumulator_mw.index):
                self._national_pv_accumulator_mw += df_for_example
            else:
                # This is the start of a new set of GSPs. First, log the national PV error:
                self._log_national_pv_error(pl_module)
                if self._national_batch_idx < 12:
                    self._plot_national_pv(pl_module.logger.experiment)
                # And reset:
                self._national_pv_accumulator_mw = df_for_example
                self._national_batch_idx += 1

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        self._log_national_pv_error(pl_module, on_step=False)

    def _log_national_pv_error(self, logger, on_step: bool = True) -> None:
        self._check_accumulator()
        error_mw = (
            self._national_pv_accumulator_mw["actual"]
            - self._national_pv_accumulator_mw["predicted"]
        )
        logger.log_dict(
            {
                "validation/national_pv_mae_mw": error_mw.abs().mean(),
                "validation/national_pv_mae_mw_2h": error_mw[:4].abs().mean(),
                "validation/national_pv_mbe_mw": error_mw.mean(),
            },
            on_epoch=True,
            on_step=on_step,
        )

    def _plot_national_pv(self, logger) -> None:
        self._check_accumulator()
        fig, ax = plt.subplots()
        ax = self._national_pv_accumulator_mw.plot(ax=ax)
        ax.set_title("National solar PV")
        ax.set_ylabel("MW")
        date = self._national_pv_accumulator_mw.index[0].date()
        ax.set_xlabel(date)
        logger.log({f"validation/national_pv_{self._national_batch_idx}": wandb.Image(fig)})
        plt.close(fig)

    def _check_accumulator(self) -> None:
        assert self._national_pv_accumulator_mw is not None
        assert len(self._national_pv_accumulator_mw) > 0
