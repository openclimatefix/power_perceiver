# General imports
import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path

# ML imports
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_forecasting.optim import Ranger
from pytorch_lightning.loggers import WandbLogger
from pytorch_msssim import ms_ssim

from power_perceiver.analysis.plot_satellite import LogSatellitePlots

# power_perceiver imports
from power_perceiver.consts import (
    NUM_HIST_SAT_IMAGES,
    X_OSGB_MEAN,
    X_OSGB_STD,
    Y_OSGB_MEAN,
    Y_OSGB_STD,
    BatchKey,
)
from power_perceiver.data_loader import HRVSatellite
from power_perceiver.data_loader.satellite import SAT_MEAN, SAT_STD
from power_perceiver.data_loader.satellite_zarr_dataset import SatelliteZarrDataset, worker_init_fn
from power_perceiver.dataset import NowcastingDataset
from power_perceiver.np_batch_processor.topography import Topography
from power_perceiver.pytorch_modules.satellite_predictor import XResUNet

logging.basicConfig()
_log = logging.getLogger("power_perceiver")
_log.setLevel(logging.DEBUG)

plt.rcParams["figure.figsize"] = (18, 10)
plt.rcParams["figure.facecolor"] = "white"

if socket.gethostname() == "donatello":
    SATELLITE_ZARR_PATH = (
        "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
        "satellite/EUMETSAT/SEVIRI_RSS/zarr/v3/eumetsat_seviri_hrv_uk.zarr"
    )
    DATA_PATH = Path(
        "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
        "prepared_ML_training_data/v15"
    )
else:
    # On Google Cloud VM:
    SATELLITE_ZARR_PATH = (
        "gs://solar-pv-nowcasting-data/satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
    )
    DATA_PATH = Path("/home/jack/data/v15")

assert DATA_PATH.exists()
NUM_FUTURE_SAT_IMAGES = 24


torch.manual_seed(42)

train_dataloader = torch.utils.data.DataLoader(
    SatelliteZarrDataset(satellite_zarr_path=SATELLITE_ZARR_PATH),
    batch_size=32,
    num_workers=1,  # TODO: Change?
    pin_memory=True,
    worker_init_fn=worker_init_fn,
    persistent_workers=True,
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
    persistent_workers=True,
)


def get_osgb_coords_for_coord_conv(batch: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
    """Returns tensor of shape (example, 2, y, x)."""
    y_osgb = batch[BatchKey.hrvsatellite_y_osgb]
    x_osgb = batch[BatchKey.hrvsatellite_x_osgb]

    # Normalise:
    y_osgb = (y_osgb - Y_OSGB_MEAN) / Y_OSGB_STD
    x_osgb = (x_osgb - X_OSGB_MEAN) / X_OSGB_STD

    # Concat:
    return torch.stack((y_osgb, x_osgb), dim=1)


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class FullModel(pl.LightningModule):
    coord_conv: bool = True
    crop: bool = False
    optimizer_class: torch.optim.Optimizer = Ranger
    optimizer_kwargs: dict = field(
        # lambda trick from https://stackoverflow.com/a/52064202/732596
        default_factory=lambda: dict(lr=1e-4)
    )
    use_topography: bool = True

    # kwargs to fastai DynamicUnet. See this page for details:
    # https://fastai1.fast.ai/vision.models.unet.html#DynamicUnet
    pretrained: bool = False
    blur_final: bool = False  # Blur final layer. fastai default is True.
    self_attention: bool = True  # Use SA layer at the third block before the end.
    last_cross: bool = True  # Use a cross-connection with the direct input of the model.
    bottle: bool = False  # Bottleneck the last skip connection.

    def __post_init__(self):
        super().__init__()

        if self.use_topography:
            self.topography = Topography("/home/jack/europe_dem_2km_osgb.tif")

        self.satellite_predictor = XResUNet(
            img_size=(64, 64),
            n_in=NUM_HIST_SAT_IMAGES + (2 if self.coord_conv else 0) + int(self.use_topography),
            n_out=NUM_FUTURE_SAT_IMAGES,
            pretrained=self.pretrained,
            blur_final=self.blur_final,
            self_attention=self.self_attention,
            last_cross=self.last_cross,
            bottle=self.bottle,
        )

        # Do this at the end of __post_init__ to capture model topology to wandb:
        self.save_hyperparameters()

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> dict[str, torch.Tensor]:

        data = x[BatchKey.hrvsatellite][:, :NUM_HIST_SAT_IMAGES, 0]
        # `data` is now of shape: (example, time, y, x)
        if self.coord_conv:
            osgb_coords = get_osgb_coords_for_coord_conv(x)
            data = torch.concat((data, osgb_coords), dim=1)

        if self.use_topography:
            x = self.topography(x)
            surface_height = x[BatchKey.hrvsatellite_surface_height]
            surface_height = surface_height.unsqueeze(1)  # Add channel dim
            data = torch.concat((data, surface_height), dim=1)

        predicted_sat = self.satellite_predictor(data)
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
            data_range=1023.0,
            size_average=True,  # Return a scalar.
            win_size=3,  # ClimateHack folks used win_size=3.
        )
        self.log(f"{tag}/ms_ssim", ms_ssim_loss)
        self.log(f"{tag}/ms_ssim+sat_mse", ms_ssim_loss + sat_mse_loss)

        if self.crop:
            # Loss on 33x33 central crop:
            # The image has to be larger than 32x32 otherwise ms-ssim complains:
            # "Image size should be larger than 32 due to the 4 downsamplings in ms-ssim"
            CROP = 15
            sat_mse_loss_crop = F.mse_loss(
                predicted_sat[:, :, CROP:-CROP, CROP:-CROP],
                actual_sat[:, :, CROP:-CROP, CROP:-CROP],
            )
            self.log(f"{tag}/sat_mse_crop", sat_mse_loss_crop)
            ms_ssim_loss_crop = 1 - ms_ssim(
                predicted_sat_denorm[:, :, CROP:-CROP, CROP:-CROP],
                actual_sat_denorm[:, :, CROP:-CROP, CROP:-CROP],
                data_range=1023.0,
                size_average=True,  # Return a scalar.
                win_size=3,  # ClimateHack folks used win_size=3.
            )
            self.log(f"{tag}/ms_ssim_crop", ms_ssim_loss_crop)
            self.log(f"{tag}/ms_ssim_crop+sat_mse_crop", ms_ssim_loss_crop + sat_mse_loss_crop)
            loss = ms_ssim_loss_crop + sat_mse_loss_crop
        else:
            loss = ms_ssim_loss + sat_mse_loss

        return dict(
            loss=loss,
            predicted_sat=predicted_sat,
            actual_sat=actual_sat,
        )

    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), **self.optimizer_kwargs)
        return optimizer


model = FullModel()

wandb_logger = WandbLogger(
    name="022.10: Topography. Ranger LR=1e-4. GCP-3",
    project="power_perceiver",
    entity="openclimatefix",
    log_model="all",
)

# log model only if validation loss decreases
if model.crop:
    loss_name = "validation/ms_ssim_crop+sat_mse_crop"
else:
    loss_name = "validation/ms_ssim+sat_mse"
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=loss_name, mode="min")

trainer = pl.Trainer(
    gpus=[0],
    max_epochs=70,
    logger=wandb_logger,
    callbacks=[
        LogSatellitePlots(),
        checkpoint_callback,
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
    ],
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
