# General imports
from dataclasses import dataclass
from pathlib import Path

# ML imports
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torch.utils import data

from power_perceiver.analysis.plot_satellite import LogSatellitePlots

# power_perceiver imports
from power_perceiver.consts import BatchKey
from power_perceiver.data_loader import HRVSatellite
from power_perceiver.dataset import NowcastingDataset
from power_perceiver.pytorch_modules.satellite_predictor import XResUNet

plt.rcParams["figure.figsize"] = (18, 10)
plt.rcParams["figure.facecolor"] = "white"

DATA_PATH = Path("/home/jack/data/v15")
#  "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
#  "prepared_ML_training_data/v15/"
assert DATA_PATH.exists()

T0_IDX_5_MIN_VALIDATION = 12
NUM_HIST_SAT_IMAGES = 7


def get_dataloader(data_path: Path, tag: str) -> data.DataLoader:
    assert tag in ["train", "validation"]
    assert data_path.exists()

    dataset = NowcastingDataset(
        data_path=data_path,
        data_loaders=[
            HRVSatellite(),
        ],
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=12,
        pin_memory=True,
    )

    return dataloader


train_dataloader = get_dataloader(DATA_PATH / "train", tag="train")
val_dataloader = get_dataloader(DATA_PATH / "test", tag="validation")


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class FullModel(pl.LightningModule):
    def __post_init__(self):
        super().__init__()

        self.satellite_predictor = XResUNet(
            input_size=(64, 64),
            history_steps=NUM_HIST_SAT_IMAGES,
            forecast_steps=24,
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
        network_out = self(batch)
        predicted_sat = network_out["predicted_sat"]
        actual_sat = batch[BatchKey.hrvsatellite][:, NUM_HIST_SAT_IMAGES:, 0]
        print(f"{predicted_sat.shape=}, {actual_sat.shape=}")
        sat_mse_loss = F.mse_loss(predicted_sat, actual_sat)
        return dict(
            loss=sat_mse_loss,
            predicted_sat=predicted_sat,
            actual_sat=actual_sat,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = FullModel()

wandb_logger = WandbLogger(
    name="021.00: ClimateHack satellite predictor",
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
