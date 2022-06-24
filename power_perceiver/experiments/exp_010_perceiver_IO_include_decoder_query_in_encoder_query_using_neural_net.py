# General imports
from dataclasses import dataclass
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import pytorch_lightning as pl

# ML imports
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils import data

from power_perceiver.analysis.plot_timeseries import LogTimeseriesPlots
from power_perceiver.analysis.plot_tsne import LogTSNEPlot
from power_perceiver.consts import BatchKey
from power_perceiver.load_prepared_batches.data_sources import PV, HRVSatellite, Sun

# power_perceiver imports
from power_perceiver.load_prepared_batches.prepared_dataset import PreparedDataset
from power_perceiver.np_batch_processor import EncodeSpaceTime, Topography
from power_perceiver.pytorch_modules.query_generator import QueryGenerator
from power_perceiver.pytorch_modules.satellite_processor import HRVSatelliteProcessor
from power_perceiver.pytorch_modules.self_attention import PerceiverIO
from power_perceiver.transforms.pv import PVPowerRollingWindow
from power_perceiver.transforms.satellite import PatchSatellite
from power_perceiver.xr_batch_processor import (
    ReduceNumPVSystems,
    ReduceNumTimesteps,
    SelectPVSystemsNearCenterOfImage,
)

plt.rcParams["figure.figsize"] = (18, 10)
plt.rcParams["figure.facecolor"] = "white"


DATA_PATH = Path(
    "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
    "prepared_ML_training_data/v15/"
)
assert DATA_PATH.exists()


def get_dataloader(data_path: Path, tag: str) -> data.DataLoader:
    assert tag in ["train", "validation"]
    assert data_path.exists()

    xr_batch_processors = [
        SelectPVSystemsNearCenterOfImage(),
        ReduceNumPVSystems(requested_num_pv_systems=8),
    ]

    if tag == "train":
        xr_batch_processors.append(ReduceNumTimesteps(requested_timesteps=4))

    dataset = PreparedDataset(
        data_path=data_path,
        data_loaders=[
            HRVSatellite(transforms=[PatchSatellite()]),
            PV(transforms=[PVPowerRollingWindow()]),
            Sun(),
        ],
        xr_batch_processors=xr_batch_processors,
        np_batch_processors=[
            EncodeSpaceTime(),
            Topography("/home/jack/europe_dem_2km_osgb.tif"),
        ],
    )

    dataloader = data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=16,
        pin_memory=True,
    )

    return dataloader


train_dataloader = get_dataloader(DATA_PATH / "train", tag="train")
val_dataloader = get_dataloader(DATA_PATH / "test", tag="validation")


def maybe_pad_with_zeros(tensor: torch.Tensor, requested_dim: int) -> torch.Tensor:
    num_zeros_to_pad = requested_dim - tensor.shape[-1]
    assert num_zeros_to_pad >= 0, f"{requested_dim=}, {tensor.shape=}"
    if num_zeros_to_pad > 0:
        zero_padding_shape = tensor.shape[:2] + (num_zeros_to_pad,)
        zero_padding = torch.zeros(*zero_padding_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.concat((tensor, zero_padding), dim=2)
    return tensor


@dataclass(
    eq=False
)  # See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
class Model(pl.LightningModule):
    encoder_query_dim: int = 64
    decoder_query_dim: int = (
        36  # decoder_query will be automatically padded with zeros to get to this size.
    )
    num_fourier_features: int = 16  # TOTAL for both x and y
    pv_system_id_embedding_dim: int = 16
    byte_array_dim: int = 35
    num_encoder_heads: int = 8
    num_decoder_heads: int = 6
    dropout: float = 0.0
    share_weights_across_latent_transformer_layers: bool = False
    num_latent_transformer_encoders: int = 4

    # Other params:
    num_elements_query_padding: int = (
        0  # Probably keep this at zero while using MultiLayerTransformerEncoder or Perceiver IO
    )

    def __post_init__(self):
        super().__init__()
        self.hrvsatellite_processor = HRVSatelliteProcessor()

        self.decoder_query_generator = QueryGenerator(
            num_fourier_features=self.num_fourier_features,  # TOTAL (for both x and y)
            pv_system_id_embedding_dim=self.pv_system_id_embedding_dim,
            num_elements_query_padding=self.num_elements_query_padding,
        )

        self.encoder_query_generator = nn.Sequential(
            nn.Linear(in_features=self.decoder_query_generator.query_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=self.encoder_query_dim),
        )

        self.perceiver_io = PerceiverIO(
            encoder_query_dim=self.encoder_query_dim,
            decoder_query_dim=self.decoder_query_dim,
            byte_array_dim=self.byte_array_dim,
            num_encoder_heads=self.num_encoder_heads,
            num_decoder_heads=self.num_decoder_heads,
            dropout=self.dropout,
            share_weights_across_latent_transformer_layers=(
                self.share_weights_across_latent_transformer_layers
            ),
            num_latent_transformer_encoders=self.num_latent_transformer_encoders,
        )

        self.output_module = nn.Sequential(
            nn.Linear(in_features=self.decoder_query_dim, out_features=self.decoder_query_dim),
            nn.ReLU(),
            nn.Linear(in_features=self.decoder_query_dim, out_features=1),
        )

        # Do this at the end of __post_init__ to capture model topology to wandb:
        self.save_hyperparameters()

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        original_batch_size = x[BatchKey.pv].shape[0]
        byte_array = self.hrvsatellite_processor(x)

        # Get decoder query
        decoder_query = self.decoder_query_generator(x)
        # Pad with zeros if necessary to get up to self.decoder_query_dim:
        decoder_query_padded = maybe_pad_with_zeros(
            decoder_query, requested_dim=self.decoder_query_dim
        )

        # Generate encoder query
        encoder_query = self.encoder_query_generator(decoder_query)

        # Run through the Perceiver IO:
        out = self.perceiver_io(
            encoder_query=encoder_query,
            byte_array=byte_array,
            decoder_query=decoder_query_padded,
        )

        out = self.output_module(out)

        # Reshape back to (batch_size, n_timesteps, ...)
        return einops.rearrange(
            out,
            "(batch_size n_timesteps) ... -> batch_size n_timesteps ...",
            batch_size=original_batch_size,
        )

    def _training_or_validation_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int, tag: str
    ) -> dict[str, object]:
        """
        Args:
            batch: The training or validation batch.  A dictionary.
            tag: Either "train" or "validation"
            batch_idx: The index of the batch.
        """
        actual_pv_power = batch[BatchKey.pv]
        actual_pv_power = torch.where(
            batch[BatchKey.pv_mask].unsqueeze(1),
            actual_pv_power,
            torch.tensor(0.0, dtype=actual_pv_power.dtype, device=actual_pv_power.device),
        )

        predicted_pv_power = self(batch).squeeze()
        mse_loss = F.mse_loss(predicted_pv_power, actual_pv_power)

        self.log(f"{tag}/mse", mse_loss)

        return {
            "loss": mse_loss,
            "predicted_pv_power": predicted_pv_power,
        }

    def training_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        return self._training_or_validation_step(batch=batch, batch_idx=batch_idx, tag="train")

    def validation_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        return self._training_or_validation_step(batch=batch, batch_idx=batch_idx, tag="validation")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


model = Model()

wandb_logger = WandbLogger(
    project="power_perceiver",
    entity="openclimatefix",
    log_model="all",
)

# log gradients, parameter histogram and model topology
wandb_logger.watch(model, log="all")

trainer = pl.Trainer(
    gpus=[5],
    max_epochs=-1,
    logger=wandb_logger,
    callbacks=[
        LogTimeseriesPlots(),
        LogTSNEPlot(),
    ],
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
