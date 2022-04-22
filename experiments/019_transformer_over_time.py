# General imports
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

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

# power_perceiver imports
from power_perceiver.analysis.plot_tsne import LogTSNEPlot
from power_perceiver.consts import BatchKey
from power_perceiver.data_loader import GSP, PV, HRVSatellite, Sun
from power_perceiver.dataset import NowcastingDataset
from power_perceiver.np_batch_processor import EncodeSpaceTime, Topography
from power_perceiver.pytorch_modules.query_generator import GSPQueryGenerator, PVQueryGenerator
from power_perceiver.pytorch_modules.satellite_processor import HRVSatelliteProcessor
from power_perceiver.pytorch_modules.self_attention import MultiLayerTransformerEncoder
from power_perceiver.transforms.pv import PVPowerRollingWindow
from power_perceiver.transforms.satellite import PatchSatellite
from power_perceiver.xr_batch_processor import ReduceNumPVSystems, SelectPVSystemsNearCenterOfImage

#  from power_perceiver.analysis.plot_timeseries import LogTimeseriesPlots


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

    dataset = NowcastingDataset(
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
class InferSingleTimestepOfPower(pl.LightningModule):
    """Infers a single timestep of PV power and GSP power at a time.

    Currently just uses HRV satellite imagery as the input. In the near future it could also
    use NWP temperature, wind speed & precipitation, and absolute geo position.
    """

    # Params for Perceiver
    # byte_array and query will be automatically padded with zeros to get to d_model.
    # Set d_model to be divisible by `num_heads`.
    d_model: int = 96
    pv_system_id_embedding_dim: int = 16
    num_heads: int = 12
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

        self.pv_query_generator = PVQueryGenerator(pv_system_id_embedding=id_embedding)

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

        self.pv_output_module = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
            nn.GELU(),
            nn.Linear(in_features=self.d_model, out_features=1),
            nn.ReLU(),  # ReLU output guarantees that we can't predict negative PV power!
        )

    def forward(
        self,
        x: dict[BatchKey, torch.Tensor],
        start_idx_5_min: Optional[int] = None,
    ) -> dict[str, torch.Tensor]:
        if start_idx_5_min is None:
            assert self.training
            # Jitter the start_idx during training.
            device = x[BatchKey.pv].device
            start_idx_5_min = torch.randint(low=0, high=22, size=(1,), device=device)[0]

        start_idx_5_min_tensor = torch.tensor(start_idx_5_min)
        gsp_time_idx_30_min = (start_idx_5_min_tensor / 6).ceil().int() + 1

        byte_array = self.hrvsatellite_processor(x, start_idx_5_min=start_idx_5_min)
        pv_query = self.pv_query_generator(x, start_idx_5_min=start_idx_5_min)
        gsp_query = self.gsp_query_generator(x, time_idx_30_min=gsp_time_idx_30_min)

        # Pad with zeros if necessary to get up to self.d_model:
        byte_array = maybe_pad_with_zeros(byte_array, requested_dim=self.d_model)
        pv_query = maybe_pad_with_zeros(pv_query, requested_dim=self.d_model)
        gsp_query = maybe_pad_with_zeros(gsp_query, requested_dim=self.d_model)

        # Prepare the attention input and run through the transformer_encoder:
        attn_input = torch.concat((pv_query, gsp_query, byte_array), dim=1)
        attn_output = attn_input + self.transformer_encoder(attn_input)

        # Select the elements of the output which correspond to the query:
        gsp_start_idx = pv_query.shape[1]
        gsp_end_idx = gsp_start_idx + gsp_query.shape[1]
        pv_attn_out = attn_output[:, :gsp_start_idx]
        gsp_attn_out = attn_output[:, gsp_start_idx:gsp_end_idx]
        predicted_pv_power = self.pv_output_module(pv_attn_out)
        predicted_gsp_power = self.pv_output_module(gsp_attn_out)

        return {
            "pv_attn_out": pv_attn_out,  # shape: (example, n_pv_systems, d_model)
            "gsp_attn_out": gsp_attn_out,  # shape: (example, 1, d_model)
            "predicted_pv_power": predicted_pv_power,  # shape: (example, n_pv_systems=8, 1)
            "predicted_gsp_power": predicted_gsp_power,  # shape: (example, 1, 1)
            "start_idx_5_min": start_idx_5_min,
            "gsp_time_idx_30_min": gsp_time_idx_30_min,
        }


def get_multi_timestep_prediction(
    model: nn.Module,
    batch: dict[BatchKey, torch.Tensor],
    start_idxs_5_min: Iterable[int],
) -> dict[str, torch.Tensor]:
    predicted_pv_powers = []
    predicted_gsp_powers = []
    actual_gsp_powers = []
    gsp_times_utc = []
    pv_attn_outs = []
    gsp_attn_outs = []
    for start_idx_5_min in start_idxs_5_min:
        out = model.forward(batch, start_idx_5_min=start_idx_5_min)
        predicted_pv_powers.append(out["predicted_pv_power"])
        predicted_gsp_powers.append(out["predicted_gsp_power"][:, :, 0])
        gsp_time_idx_30_min = out["gsp_time_idx_30_min"]
        actual_gsp_power = batch[BatchKey.gsp][:, gsp_time_idx_30_min]  # Shape: (example)
        actual_gsp_power = actual_gsp_power.unsqueeze(1)  # Shape: (example, time=1)
        actual_gsp_powers.append(actual_gsp_power)
        gsp_time_utc = batch[BatchKey.gsp_time_utc][:, gsp_time_idx_30_min]
        gsp_time_utc = gsp_time_utc.unsqueeze(1)  # Shape: (example, time=1)
        gsp_times_utc.append(gsp_time_utc)
        pv_attn_outs.append(out["pv_attn_out"].unsqueeze(1))  # Shape: (example, 1, n_pv, d_mod)
        gsp_attn_outs.append(out["gsp_attn_out"])

    predicted_pv_power = torch.concat(predicted_pv_powers, dim=2)
    predicted_gsp_power = torch.concat(predicted_gsp_powers, dim=1)
    actual_gsp_power = torch.concat(actual_gsp_powers, dim=1)
    gsp_time_utc = torch.concat(gsp_times_utc, dim=1)
    pv_attn_out = torch.concat(pv_attn_outs, dim=1)
    gsp_attn_out = torch.concat(gsp_attn_outs, dim=1)

    del predicted_pv_powers, predicted_gsp_powers, actual_gsp_powers, gsp_times_utc
    del pv_attn_outs, gsp_attn_outs

    predicted_pv_power = einops.rearrange(
        predicted_pv_power,
        "example n_pv_systems time -> example time n_pv_systems",
        n_pv_systems=8,  # sanity check!
    )
    return dict(
        pv_attn_out=pv_attn_out,  # Shape: (example, time, n_pv_systems, d_model)
        gsp_attn_out=gsp_attn_out,  # Shape: (example, time, d_model)
        predicted_pv_power=predicted_pv_power,  # Shape: (example, time, n_pv_systems)
        predicted_gsp_power=predicted_gsp_power,  # Shape: (example, time)
        actual_gsp_power=actual_gsp_power,  # Shape: (example, time)
        gsp_time_utc=gsp_time_utc,  # Shape: (example, time)
    )


class TrainOrValidationMixIn:
    def _training_or_validation_step(
        self,
        batch: dict[BatchKey, torch.Tensor],
        batch_idx: int,
        tag: str,
        predicted_pv_power,
        predicted_gsp_power,
        actual_pv_power,
        actual_gsp_power,
        gsp_time_utc,
    ) -> dict[str, object]:
        """
        Args:
            batch: The training or validation batch.  A dictionary.
            tag: Either "train" or "validation"
            batch_idx: The index of the batch.
        """
        # Set PV power to zero for missing PV systems:
        actual_pv_power = torch.where(
            batch[BatchKey.pv_mask].unsqueeze(1),
            actual_pv_power,
            torch.tensor(0.0, dtype=actual_pv_power.dtype, device=actual_pv_power.device),
        )

        # PV power loss:
        pv_mse_loss = F.mse_loss(predicted_pv_power, actual_pv_power)
        pv_nmae_loss = F.l1_loss(predicted_pv_power, actual_pv_power)
        self.log(f"{tag}/pv_mse", pv_mse_loss)
        self.log(f"{tag}/pv_nmae", pv_nmae_loss)
        self.log(f"{tag}/mse", pv_mse_loss)  # To allow for each comparison to older models.

        # GSP power loss:
        gsp_mse_loss = F.mse_loss(predicted_gsp_power, actual_gsp_power)
        gsp_nmae_loss = F.l1_loss(predicted_gsp_power, actual_gsp_power)
        self.log(f"{tag}/gsp_mse", gsp_mse_loss)
        self.log(f"{tag}/gsp_nmae", gsp_nmae_loss)

        # Total MSE loss:
        total_mse_loss = pv_mse_loss + gsp_mse_loss
        self.log(f"{tag}/total_mse", total_mse_loss)

        return {
            "loss": total_mse_loss,
            "pv_mse_loss": pv_mse_loss,
            "gsp_mse_loss": gsp_mse_loss,
            "pv_nmae_loss": pv_nmae_loss,
            "gsp_nmae_loss": gsp_nmae_loss,
            "predicted_gsp_power": predicted_gsp_power,
            "actual_gsp_power": actual_gsp_power,
            "gsp_time_utc": gsp_time_utc,
            "predicted_pv_power": predicted_pv_power,
        }


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class TrainInferSingleTimestepOfPower(pl.LightningModule, TrainOrValidationMixIn):
    def __post_init__(self):
        super().__init__()
        self.infer_single_timestep_of_power = InferSingleTimestepOfPower.load_from_checkpoint(
            "/home/jack/dev/ocf/power_perceiver/experiments/model_checkpoints/018.13/model.ckpt"
        )

        # Do this at the end of __post_init__ to capture model topology to wandb:
        self.save_hyperparameters()

    def forward(
        self, x: dict[BatchKey, torch.Tensor], start_idx_5_min: Optional[int] = None
    ) -> dict[str, torch.Tensor]:
        return self.infer_single_timestep_of_power(x=x, start_idx_5_min=start_idx_5_min)

    def training_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        """Train on a single (random) timestep per example."""
        out = self.forward(batch)
        start_idx_5_min = out["start_idx_5_min"]
        # Select PV data:
        # We want the PV data that is 30 minutes into the 45 min sequence of images:
        actual_pv_power = batch[BatchKey.pv][:, 6 + start_idx_5_min]
        actual_pv_power = actual_pv_power.unsqueeze(1)  # Shape: (example, time=1, pv_systems)
        predicted_pv_power = out["predicted_pv_power"]
        predicted_pv_power = einops.rearrange(
            predicted_pv_power,
            "example n_pv_systems 1 -> example 1 n_pv_systems",
            n_pv_systems=actual_pv_power.shape[2],  # sanity check!
        )
        # Select a single timestep of GSP data:
        gsp_time_idx_30_min = out["gsp_time_idx_30_min"]
        actual_gsp_power = batch[BatchKey.gsp][:, gsp_time_idx_30_min]  # Shape: (example)
        actual_gsp_power = actual_gsp_power.unsqueeze(1)  # Shape: (example, time=1)
        predicted_gsp_power = out["predicted_gsp_power"][:, :, 0]  # Shape: (example, time=1)
        gsp_time_utc = batch[BatchKey.gsp_time_utc][:, gsp_time_idx_30_min]
        gsp_time_utc = gsp_time_utc.unsqueeze(1)  # Shape: (example, time=1)
        return self._training_or_validation_step(
            batch=batch,
            batch_idx=batch_idx,
            tag="train",
            predicted_pv_power=predicted_pv_power,
            predicted_gsp_power=predicted_gsp_power,
            actual_pv_power=actual_pv_power,
            actual_gsp_power=actual_gsp_power,
            gsp_time_utc=gsp_time_utc,
        )

    def validation_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        """Validate on multiple timesteps."""
        multi_timestep_prediction = get_multi_timestep_prediction(
            model=self, batch=batch, start_idxs_5_min=range(0, 22)
        )
        for key in ("pv_attn_out", "gsp_attn_out"):
            del multi_timestep_prediction[key]
        actual_pv_power = batch[BatchKey.pv][:, 6:-3]  # example, time, n_pv_systems
        return self._training_or_validation_step(
            batch=batch,
            batch_idx=batch_idx,
            tag="validation",
            actual_pv_power=actual_pv_power,
            **multi_timestep_prediction,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class FullModel(pl.LightningModule, TrainOrValidationMixIn):
    d_model: int = 96  # Must be the same as for InferSingleTimestepOfPower
    pv_system_id_embedding_dim: int = 16
    num_heads: int = 12
    dropout: float = 0.0
    share_weights_across_latent_transformer_layers: bool = False
    num_latent_transformer_encoders: int = 4

    def __post_init__(self):
        super().__init__()
        self.infer_single_timestep_of_power = InferSingleTimestepOfPower.load_from_checkpoint(
            "/home/jack/dev/ocf/power_perceiver/experiments/model_checkpoints/018.13/model.ckpt"
        )

        assert self.d_model == self.infer_single_timestep_of_power.d_model

        self.time_transformer_encoder = MultiLayerTransformerEncoder(
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
        multi_timestep_prediction = get_multi_timestep_prediction(
            model=self.infer_single_timestep_of_power,
            batch=x,
            start_idxs_5_min=range(3, 22, 6),  # Every half hour.
        )
        """
        pv_attn_out=pv_attn_out,  # Shape: (example, time, n_pv_systems, d_model)
        gsp_attn_out=gsp_attn_out,  # Shape: (example, time, d_model)
        predicted_pv_power=predicted_pv_power,  # Shape: (example, time, n_pv_systems)
        predicted_gsp_power=predicted_gsp_power,  # Shape: (example, time)
        actual_gsp_power=actual_gsp_power,  # Shape: (example, time)
        gsp_time_utc=gsp_time_utc,  # Shape: (example, time)
        """

        pv_attn_out = multi_timestep_prediction["pv_attn_out"]
        n_timesteps, n_pv_systems = pv_attn_out.shape[1:3]
        pv_attn_out = einops.rearrange(
            pv_attn_out,
            "example time n_pv_systems d_model -> example (time n_pv_systems) d_model",
        )
        num_pv_elements = pv_attn_out.shape[1]
        gsp_attn_out = multi_timestep_prediction["gsp_attn_out"]

        # TODO: Add in historical PV :)
        time_attn_in = torch.concat((pv_attn_out, gsp_attn_out), dim=1)
        time_attn_out = self.time_transformer_encoder(time_attn_in)

        power_out = self.pv_output_module(time_attn_out)  # (example, total_num_elements, 1)

        # Reshape the PV power predictions
        predicted_pv_power = power_out[:, :num_pv_elements]
        predicted_pv_power = einops.rearrange(
            predicted_pv_power,
            "example (time n_pv_systems) 1 -> example time n_pv_systems",
            time=n_timesteps,
            n_pv_systems=n_pv_systems,
        )

        # Shape: (example time n_pv_systems)
        multi_timestep_prediction["predicted_pv_power"] = predicted_pv_power
        # Shape: (example, time)
        multi_timestep_prediction["predicted_gsp_power"] = power_out[:, num_pv_elements:, 0]

        return multi_timestep_prediction

    def training_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        return self.validation_step(batch, batch_idx)

    def validation_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        """Validate on multiple timesteps."""
        multi_timestep_prediction = self(batch)
        for key in ("pv_attn_out", "gsp_attn_out"):
            del multi_timestep_prediction[key]
        actual_pv_power = batch[BatchKey.pv][:, 6:-3:6]  # example, time, n_pv_systems
        return self._training_or_validation_step(
            batch=batch,
            batch_idx=batch_idx,
            tag="train" if self.training else "validation",
            actual_pv_power=actual_pv_power,
            **multi_timestep_prediction,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer


model = FullModel()

wandb_logger = WandbLogger(
    name="019.04: Implement Time Transformer",
    project="power_perceiver",
    entity="openclimatefix",
    log_model="all",
)

# log gradients, parameter histogram and model topology
wandb_logger.watch(model, log="all")

trainer = pl.Trainer(
    gpus=[4],
    max_epochs=70,
    logger=wandb_logger,
    callbacks=[
        # LogTimeseriesPlots(),
        LogTSNEPlot(query_generator_name="infer_single_timestep_of_power.pv_query_generator"),
    ],
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
