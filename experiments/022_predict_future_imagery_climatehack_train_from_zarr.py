# General imports
from dataclasses import dataclass
from pathlib import Path

# ML imports
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_msssim import ms_ssim

from power_perceiver.analysis.plot_satellite import LogSatellitePlots

# power_perceiver imports
from power_perceiver.consts import NUM_HIST_SAT_IMAGES, BatchKey
from power_perceiver.data_loader import HRVSatellite
from power_perceiver.data_loader.satellite import SAT_MEAN, SAT_STD
from power_perceiver.data_loader.satellite_from_zarr import SatelliteDataset, worker_init_fn
from power_perceiver.dataset import NowcastingDataset
from power_perceiver.pytorch_modules.satellite_predictor import XResUNet

plt.rcParams["figure.figsize"] = (18, 10)
plt.rcParams["figure.facecolor"] = "white"

SATELLITE_ZARR_PATH = (
    # "gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
    "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
    "satellite/EUMETSAT/SEVIRI_RSS/zarr/v3/eumetsat_seviri_hrv_uk.zarr"
)
DATA_PATH = Path("/home/jack/data/v15")
#  "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
#  "prepared_ML_training_data/v15/"
assert DATA_PATH.exists()
NUM_FUTURE_SAT_IMAGES = 24


torch.manual_seed(42)

train_dataloader = torch.utils.data.DataLoader(
    SatelliteDataset(satellite_zarr_path=SATELLITE_ZARR_PATH),
    batch_size=32,
    num_workers=1,  # TODO: Change?
    pin_memory=True,
    worker_init_fn=worker_init_fn,
)

val_dataloader = torch.utils.data.DataLoader(
    NowcastingDataset(
        data_path=DATA_PATH / "test",
        data_loaders=[
            HRVSatellite(),
        ],
    ),
    batch_size=None,
    num_workers=12,
    pin_memory=True,
)


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class FullModel(pl.LightningModule):
    def __post_init__(self):
        super().__init__()

        self.satellite_predictor = XResUNet(
            input_size=(64, 64),
            history_steps=NUM_HIST_SAT_IMAGES,
            forecast_steps=NUM_FUTURE_SAT_IMAGES,
            pretrained=False,
        )

        # Do this at the end of __post_init__ to capture model topology to wandb:
        self.save_hyperparameters()

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> dict[str, torch.Tensor]:
        historical_sat = x[BatchKey.hrvsatellite][:, :NUM_HIST_SAT_IMAGES, 0]
        predicted_sat = self.satellite_predictor(historical_sat)
        return dict(predicted_sat=predicted_sat)

    def training_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        return self.validation_step(batch, batch_idx)

    def validation_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        tag = "train" if self.training else "validation"
        network_out = self(batch)
        predicted_sat = network_out["predicted_sat"]
        actual_sat = batch[BatchKey.hrvsatellite][:, NUM_HIST_SAT_IMAGES:, 0]
        sat_mse_loss = F.mse_loss(predicted_sat, actual_sat)
        self.log(f"{tag}/sat_mse", sat_mse_loss)

        # MS-SSIM. Requires images to be de-normalised:
        actual_sat_denorm = (actual_sat * SAT_STD["HRV"]) + SAT_MEAN["HRV"]
        predicted_sat_denorm = (predicted_sat * SAT_STD["HRV"]) + SAT_MEAN["HRV"]
        ms_ssim_loss = 1 - ms_ssim(
            predicted_sat_denorm,
            actual_sat_denorm,
            data_range=1023,
            size_average=True,  # Return a scalar.
            win_size=3,  # ClimateHack folks used win_size=3.
        )
        self.log(f"{tag}/ms_ssim", ms_ssim_loss)
        self.log(f"{tag}/ms_ssim+sat_mse", ms_ssim_loss + sat_mse_loss)

        return dict(
            loss=ms_ssim_loss + sat_mse_loss,
            predicted_sat=predicted_sat,
            actual_sat=actual_sat,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


model = FullModel()

wandb_logger = WandbLogger(
    name=(
        "022.01: Train from sat zarr. MS-SSIM+SAT_MSE. LR=1e-4. ClimateHack satellite predictor."
        " donatello GPU 0. n_days_to_load_per_epoch=128. n_examples_per_epoch=1024x128"
    ),
    project="power_perceiver",
    entity="openclimatefix",
    log_model="all",
)

# log gradients, parameter histogram and model topology
# wandb_logger.watch(model, log="all")

trainer = pl.Trainer(
    gpus=[0],
    max_epochs=70,
    logger=wandb_logger,
    callbacks=[
        LogSatellitePlots(),
    ],
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
