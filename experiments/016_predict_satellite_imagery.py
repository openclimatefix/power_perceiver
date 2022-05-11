# General imports
from dataclasses import dataclass
from pathlib import Path

import einops

# ML imports
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.utils import data

from power_perceiver.analysis.plot_timeseries import LogTimeseriesPlots

# power_perceiver imports
from power_perceiver.analysis.plot_tsne import LogTSNEPlot
from power_perceiver.consts import BatchKey
from power_perceiver.load_prepared_batches.data_sources import PV, HRVSatellite, Sun
from power_perceiver.load_prepared_batches.prepared_dataset import PreparedDataset
from power_perceiver.np_batch_processor import EncodeSpaceTime, Topography
from power_perceiver.pytorch_modules.query_generator import QueryGenerator
from power_perceiver.pytorch_modules.satellite_processor import HRVSatelliteProcessor
from power_perceiver.pytorch_modules.self_attention import MultiLayerTransformerEncoder
from power_perceiver.transforms.pv import PVPowerRollingWindow
from power_perceiver.transforms.satellite import PatchSatellite
from power_perceiver.xr_batch_processor import ReduceNumPVSystems, SelectPVSystemsNearCenterOfImage

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

    dataset = PreparedDataset(
        data_path=data_path,
        data_loaders=[
            HRVSatellite(
                transforms=[
                    PatchSatellite(),
                ]
            ),
            PV(transforms=[PVPowerRollingWindow()]),
            Sun(),
        ],
        xr_batch_processors=[
            SelectPVSystemsNearCenterOfImage(),
            ReduceNumPVSystems(requested_num_pv_systems=8),
        ],
        np_batch_processors=[
            EncodeSpaceTime(),
            Topography("/home/jack/europe_dem_2km_osgb.tif"),
        ],
    )

    def seed_rngs(worker_id: int):
        """Set different random seed per worker."""
        worker_info = torch.utils.data.get_worker_info()
        for xr_batch_processor in worker_info.dataset.xr_batch_processors:
            if getattr(xr_batch_processor, "rng", None):
                xr_batch_processor.rng = np.random.default_rng(seed=42 + worker_id)

    dataloader = data.DataLoader(
        dataset,
        batch_size=None,
        num_workers=16,
        pin_memory=True,
        worker_init_fn=seed_rngs,
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


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class Model(pl.LightningModule):
    # Params for Perceiver
    # byte_array and query will be automatically padded with zeros to get to d_model.
    # Set d_model to be divisible by `num_heads`.
    d_model: int = 96
    byte_array_dim: int = 91
    query_dim: int = 54
    pv_system_id_embedding_dim: int = 16
    num_heads: int = 12
    dropout: float = 0.0
    share_weights_across_latent_transformer_layers: bool = False
    num_latent_transformer_encoders: int = 4

    def __post_init__(self):
        super().__init__()
        self.hrvsatellite_processor = HRVSatelliteProcessor()

        self.query_generator = QueryGenerator(
            pv_system_id_embedding_dim=self.pv_system_id_embedding_dim,
        )

        self.query_processor = nn.Sequential(
            nn.Linear(in_features=self.query_dim, out_features=self.d_model),
            nn.GELU(),
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
        )

        self.byte_array_processor = nn.Sequential(
            nn.Linear(in_features=self.byte_array_dim, out_features=self.d_model),
            nn.GELU(),
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
        )

        self.transformer_encoder = MultiLayerTransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            share_weights_across_latent_transformer_layers=(
                self.share_weights_across_latent_transformer_layers
            ),
            num_latent_transformer_encoders=self.num_latent_transformer_encoders,
        )

        self.pv_output_module = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
            nn.GELU(),
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
            nn.GELU(),
            nn.Linear(in_features=self.d_model, out_features=1),
        )

        self.imagery_output_module = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
            nn.GELU(),
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
            nn.GELU(),
            nn.Linear(in_features=self.d_model, out_features=64),
        )

        # Do this at the end of __post_init__ to capture model topology:
        self.save_hyperparameters()

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        if self.training:
            start_idx = torch.randint(low=0, high=7, size=(1,), device=x[BatchKey.pv].device)[0]
        else:  # Don't jitter during eval
            start_idx = 0
        byte_array = self.hrvsatellite_processor(x, start_idx=start_idx)
        query = self.query_generator(x, start_idx=start_idx)

        # Pad with zeros if necessary to get up to self.d_model:
        # byte_array = maybe_pad_with_zeros(byte_array, requested_dim=self.d_model)
        # query = maybe_pad_with_zeros(query, requested_dim=self.d_model)

        byte_array = self.byte_array_processor(byte_array)
        query = self.query_processor(query)

        # Prepare the attention input and run through the transformer_encoder:
        attn_input = torch.concat((byte_array, query), dim=1)
        attn_output = attn_input + self.transformer_encoder(attn_input)

        # Select the elements of the output which correspond to the query:
        pv_out = query[..., -1:] + self.pv_output_module(attn_output[:, byte_array.shape[1] :])
        imagery_out = byte_array[..., -64:] + self.imagery_output_module(
            attn_output[:, : byte_array.shape[1]]
        )

        return {
            "pv_out": pv_out,
            "imagery_out": imagery_out,
            "start_idx": start_idx,
        }

    def _training_or_validation_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int, tag: str
    ) -> dict[str, object]:
        """
        Args:
            batch: The training or validation batch.  A dictionary.
            tag: Either "train" or "validation"
            batch_idx: The index of the batch.
        """
        # PV prediction
        actual_pv_power = batch[BatchKey.pv]  # example, time, n_pv_systems
        out = self(batch)
        # Select just a single timestep:
        start_idx = out["start_idx"]
        actual_pv_power = actual_pv_power[:, 12 + start_idx : 24 + start_idx]
        predicted_pv_power = out["pv_out"]
        predicted_pv_power = einops.rearrange(
            predicted_pv_power,
            "example (time n_pv_systems) 1 -> example time n_pv_systems",
            time=actual_pv_power.shape[1],
            n_pv_systems=actual_pv_power.shape[2],
        )

        actual_pv_power = torch.where(
            batch[BatchKey.pv_mask].unsqueeze(1),
            actual_pv_power,
            torch.tensor(0.0, dtype=actual_pv_power.dtype, device=actual_pv_power.device),
        )

        pv_mse_loss = F.mse_loss(predicted_pv_power, actual_pv_power)
        self.log(f"{tag}/mse", pv_mse_loss)

        # Imagery
        actual_imagery = self.hrvsatellite_processor(
            batch, start_idx=start_idx + 12, satellite_only=True
        )
        predicted_imagery = out["imagery_out"]
        imagery_mse_loss = F.mse_loss(predicted_imagery, actual_imagery)
        self.log(f"{tag}/imagery_mse", imagery_mse_loss)

        total_mse_loss = pv_mse_loss + imagery_mse_loss
        self.log(f"{tag}/total_mse", total_mse_loss)

        return {
            "loss": total_mse_loss,
            "predicted_pv_power": predicted_pv_power,
            "predicted_imagery": predicted_imagery,
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer


model = Model()

wandb_logger = WandbLogger(
    name="016.06",
    project="power_perceiver",
    entity="openclimatefix",
    log_model="all",
)

# log gradients, parameter histogram and model topology
wandb_logger.watch(model, log="all")

trainer = pl.Trainer(
    gpus=[0],
    max_epochs=70,
    logger=wandb_logger,
    callbacks=[
        LogTimeseriesPlots(),
        LogTSNEPlot(query_generator_name="query_generator"),
    ],
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
