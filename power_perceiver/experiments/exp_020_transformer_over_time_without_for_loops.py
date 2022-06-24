# General imports
from copy import deepcopy
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
from power_perceiver.consts import T0_IDX_30_MIN, BatchKey
from power_perceiver.load_prepared_batches.data_sources import GSP, PV, HRVSatellite, Sun
from power_perceiver.load_prepared_batches.prepared_dataset import PreparedDataset
from power_perceiver.np_batch_processor import EncodeSpaceTime, Topography
from power_perceiver.np_batch_processor.align_gsp_to_5_min import GSP5Min
from power_perceiver.pytorch_modules.query_generator import GSPQueryGenerator, PVQueryGenerator
from power_perceiver.pytorch_modules.satellite_processor import HRVSatelliteProcessor
from power_perceiver.pytorch_modules.self_attention import MultiLayerTransformerEncoder
from power_perceiver.transforms.pv import PVPowerRollingWindow
from power_perceiver.transforms.satellite import PatchSatellite
from power_perceiver.xr_batch_processor import (
    AlignGSPTo5Min,
    ReduceNumPVSystems,
    ReduceNumTimesteps,
    SelectPVSystemsNearCenterOfImage,
)

plt.rcParams["figure.figsize"] = (18, 10)
plt.rcParams["figure.facecolor"] = "white"

DATA_PATH = Path("/home/jack/data/v15")
#  "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
#  "prepared_ML_training_data/v15/"
assert DATA_PATH.exists()

D_MODEL = 128
N_HEADS = 16
T0_IDX_5_MIN_TRAINING = 3
T0_IDX_5_MIN_VALIDATION = 12


def get_dataloader(data_path: Path, tag: str) -> data.DataLoader:
    assert tag in ["train", "validation"]
    assert data_path.exists()

    xr_batch_processors = [
        SelectPVSystemsNearCenterOfImage(),
        ReduceNumPVSystems(requested_num_pv_systems=8),
        AlignGSPTo5Min(),
    ]
    if tag == "train":
        xr_batch_processors.append(ReduceNumTimesteps(keys=(HRVSatellite, PV, GSP5Min, Sun)))

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
            GSP(),
        ],
        xr_batch_processors=xr_batch_processors,
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
        num_workers=12,
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
class SatelliteTransformer(nn.Module):
    """Infers a single timestep of PV power and GSP power at a time.

    Currently just uses HRV satellite imagery as the input. In the near future it could also
    use NWP temperature, wind speed & precipitation, and absolute geo position.
    """

    # Params for Perceiver
    # byte_array and query will be automatically padded with zeros to get to d_model.
    # Set d_model to be divisible by `num_heads`.
    d_model: int = D_MODEL
    pv_system_id_embedding_dim: int = 16
    num_heads: int = N_HEADS
    dropout: float = 0.0
    share_weights_across_latent_transformer_layers: bool = False
    num_latent_transformer_encoders: int = 4

    def __post_init__(self):
        super().__init__()
        self.hrvsatellite_processor = HRVSatelliteProcessor()

        NUM_PV_SYSTEMS = 1400
        NUM_GSPS = 360
        id_embedding = nn.Embedding(
            num_embeddings=NUM_PV_SYSTEMS + NUM_GSPS,
            embedding_dim=self.pv_system_id_embedding_dim,
        )

        self.pv_query_generator = PVQueryGenerator(
            pv_system_id_embedding=id_embedding,
            t0_idx_5_min_training=T0_IDX_5_MIN_TRAINING,
            t0_idx_5_min_validation=T0_IDX_5_MIN_VALIDATION,
        )

        self.gsp_query_generator = GSPQueryGenerator(gsp_id_embedding=id_embedding)

        self.transformer_encoder = MultiLayerTransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            share_weights_across_latent_transformer_layers=(
                self.share_weights_across_latent_transformer_layers
            ),
            num_latent_transformer_encoders=self.num_latent_transformer_encoders,
        )

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Reshape so each timestep is considered a different example:
        num_examples, num_timesteps = x[BatchKey.pv].shape[:2]
        x = deepcopy(x)  # TODO: Fix this hack, which fixes the issue
        # that our reshaping affects the time_transformer, too!
        for batch_key in (
            BatchKey.gsp_5_min_time_utc_fourier,
            BatchKey.pv_time_utc_fourier,
            BatchKey.hrvsatellite_solar_azimuth,
            BatchKey.hrvsatellite_solar_elevation,
            BatchKey.hrvsatellite_actual,
            BatchKey.hrvsatellite_time_utc_fourier,
        ):
            x[batch_key] = einops.rearrange(x[batch_key], "example time ... -> (example time) ...")

        # Process satellite data and queries:
        pv_query = self.pv_query_generator(x, for_satellite_transformer=True)
        gsp_query = self.gsp_query_generator(x, for_satellite_transformer=True)
        satellite_data = self.hrvsatellite_processor(x)

        # Pad with zeros if necessary to get up to self.d_model:
        pv_query = maybe_pad_with_zeros(pv_query, requested_dim=self.d_model)
        gsp_query = maybe_pad_with_zeros(gsp_query, requested_dim=self.d_model)
        satellite_data = maybe_pad_with_zeros(satellite_data, requested_dim=self.d_model)

        # Mask the NaN GSP queries. True or non-zero value indicates value will be ignored.
        mask = torch.concat(
            (
                torch.zeros_like(pv_query[:, :, 0]),
                torch.isnan(gsp_query[:, :, -1]),  # The time fourier features are last in dim 2.
                torch.zeros_like(satellite_data[:, :, 0]),
            ),
            dim=1,
        )

        # Now set NaN GSP queries to zero. They'll be ignored :)
        gsp_query = torch.nan_to_num(gsp_query, nan=0)

        # Prepare the attention input and run through the transformer_encoder:
        attn_input = torch.concat((pv_query, gsp_query, satellite_data), dim=1)
        attn_output = attn_input + self.transformer_encoder(attn_input, src_key_padding_mask=mask)

        # Reshape to (example time element d_model):
        attn_output = einops.rearrange(
            attn_output,
            "(example time) ... -> example time ...",
            example=num_examples,
            time=num_timesteps,
        )

        # Select the elements of the output which correspond to the query:
        gsp_start_idx = pv_query.shape[1]
        gsp_end_idx = gsp_start_idx + gsp_query.shape[1]
        pv_attn_out = attn_output[:, :, :gsp_start_idx]
        gsp_attn_out = attn_output[:, :, gsp_start_idx:gsp_end_idx]

        return {
            "pv_attn_out": pv_attn_out,  # shape: (example, n_pv_systems, d_model)
            "gsp_attn_out": gsp_attn_out,  # shape: (example, 1, d_model)
        }


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class FullModel(pl.LightningModule):
    d_model: int = D_MODEL + N_HEADS  # + N_HEADS for historical PV
    pv_system_id_embedding_dim: int = 16
    num_heads: int = N_HEADS
    dropout: float = 0.1
    share_weights_across_latent_transformer_layers: bool = False
    num_latent_transformer_encoders: int = 4

    def __post_init__(self):
        super().__init__()
        self.satellite_transformer = SatelliteTransformer()

        self.time_transformer = MultiLayerTransformerEncoder(
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
            nn.Linear(in_features=self.d_model, out_features=1),
            nn.ReLU(),  # ReLU output guarantees that we can't predict negative PV power!
        )

        # Do this at the end of __post_init__ to capture model topology to wandb:
        self.save_hyperparameters()

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> dict[str, torch.Tensor]:
        sat_trans_out = self.satellite_transformer(x)
        pv_attn_out = sat_trans_out["pv_attn_out"]  # Shape: (example time n_pv_systems d_model)
        gsp_attn_out = sat_trans_out["gsp_attn_out"]

        # Concatenate historical PV
        t0_idx_5_min = T0_IDX_5_MIN_TRAINING if self.training else T0_IDX_5_MIN_VALIDATION
        historical_pv = torch.zeros_like(x[BatchKey.pv])  # Shape: (example, time, n_pv_systems)
        historical_pv[:, : t0_idx_5_min + 1] = x[BatchKey.pv][:, : t0_idx_5_min + 1]
        historical_pv = historical_pv.unsqueeze(-1)  # Shape: (example, time, n_pv_systems, 1)
        # Now append a "marker" to indicate which timesteps are history:
        hist_pv_marker = torch.zeros_like(historical_pv)
        hist_pv_marker[:, : t0_idx_5_min + 1] = 1
        pv_attn_out = torch.concat((pv_attn_out, historical_pv, hist_pv_marker), dim=3)

        # Reshape pv and gsp attention outputs so each timestep an each pv system is
        # seen as a separate element into the `time transformer`.
        n_timesteps, n_pv_systems = pv_attn_out.shape[1:3]
        REARRANGE_STR = "example time n_pv_systems d_model -> example (time n_pv_systems) d_model"
        pv_attn_out = einops.rearrange(pv_attn_out, REARRANGE_STR)
        pv_attn_out = maybe_pad_with_zeros(pv_attn_out, requested_dim=self.d_model)
        gsp_attn_out = einops.rearrange(gsp_attn_out, REARRANGE_STR, n_pv_systems=1)
        gsp_attn_out = maybe_pad_with_zeros(gsp_attn_out, requested_dim=self.d_model)
        n_pv_elements = pv_attn_out.shape[1]

        # Get historical PV
        # pv_query_generator = self.satellite_transformer.pv_query_generator
        # historical_pv = pv_query_generator(x, for_satellite_transformer=False)
        # historical_pv = maybe_pad_with_zeros(historical_pv, requested_dim=self.d_model)

        # Get GSP query
        gsp_query_generator = self.satellite_transformer.gsp_query_generator
        gsp_query = gsp_query_generator(x, for_satellite_transformer=False)
        gsp_query = maybe_pad_with_zeros(gsp_query, requested_dim=self.d_model)

        # Concatenate all the things we're going to feed into the "time transformer":
        time_attn_in = (
            pv_attn_out,
            gsp_attn_out,
            # historical_pv,
            gsp_query,
        )
        time_attn_in = torch.concat(time_attn_in, dim=1)
        time_attn_out = self.time_transformer(time_attn_in)

        power_out = self.pv_output_module(time_attn_out)  # (example, total_num_elements, 1)

        # Reshape the PV power predictions
        predicted_pv_power = power_out[:, :n_pv_elements]
        predicted_pv_power = einops.rearrange(
            predicted_pv_power,
            "example (time n_pv_systems) 1 -> example time n_pv_systems",
            time=n_timesteps,
            n_pv_systems=n_pv_systems,
        )

        # GSP power. There's just 1 GSP.
        n_gsp_elements = gsp_query.shape[1]
        predicted_gsp_power = power_out[:, -n_gsp_elements:, 0]

        return dict(
            predicted_pv_power=predicted_pv_power,  # Shape: (example time n_pv_systems)
            predicted_gsp_power=predicted_gsp_power,  # Shape: (example, time)
        )

    def training_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        return self.validation_step(batch, batch_idx)

    def validation_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        tag = "train" if self.training else "validation"
        network_out = self(batch)
        predicted_pv_power = network_out["predicted_pv_power"]
        predicted_gsp_power = network_out["predicted_gsp_power"]
        actual_pv_power = batch[BatchKey.pv]
        actual_pv_power = torch.where(
            batch[BatchKey.pv_mask].unsqueeze(1),
            actual_pv_power,
            torch.tensor(0.0, dtype=actual_pv_power.dtype, device=actual_pv_power.device),
        )

        # PV power loss:
        pv_mse_loss = F.mse_loss(predicted_pv_power, actual_pv_power)
        t0_idx_5_min = T0_IDX_5_MIN_TRAINING if self.training else T0_IDX_5_MIN_VALIDATION

        pv_nmae_loss = F.l1_loss(
            predicted_pv_power[:, t0_idx_5_min + 1 :], actual_pv_power[:, t0_idx_5_min + 1 :]
        )
        self.log(f"{tag}/pv_mse", pv_mse_loss)
        self.log(f"{tag}/pv_nmae", pv_nmae_loss)

        # GSP power loss:
        actual_gsp_power = batch[BatchKey.gsp]
        gsp_mse_loss = F.mse_loss(predicted_gsp_power, actual_gsp_power)
        gsp_nmae_loss = F.l1_loss(
            predicted_gsp_power[:, T0_IDX_30_MIN + 1 :], actual_gsp_power[:, T0_IDX_30_MIN + 1 :]
        )
        self.log(f"{tag}/gsp_mse", gsp_mse_loss)
        self.log(f"{tag}/gsp_nmae", gsp_nmae_loss)

        # Total MSE loss:
        total_mse_loss = pv_mse_loss + gsp_mse_loss
        self.log(f"{tag}/total_mse", total_mse_loss)

        # Total NMAE loss:
        total_nmae_loss = pv_nmae_loss + gsp_nmae_loss
        self.log(f"{tag}/total_nmae", total_nmae_loss)

        return {
            "loss": total_mse_loss,
            "pv_mse_loss": pv_mse_loss,
            "gsp_mse_loss": gsp_mse_loss,
            "pv_nmae_loss": pv_nmae_loss,
            "gsp_nmae_loss": gsp_nmae_loss,
            "predicted_gsp_power": predicted_gsp_power,
            "actual_gsp_power": actual_gsp_power,
            "gsp_time_utc": batch[BatchKey.gsp_time_utc],
            "actual_pv_power": actual_pv_power,
            "predicted_pv_power": predicted_pv_power,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer


model = FullModel()

wandb_logger = WandbLogger(
    name=(
        "020.11: GCP! Mark hist. MSE objective. D_MODEL=128. 4 TT layers."
        " Concat hist PV. 12 timesteps during training. LR=5e-5"
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
        LogTimeseriesPlots(),
        LogTSNEPlot(query_generator_name="satellite_transformer.pv_query_generator"),
    ],
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
