"""
U-Net followed by the Power Perceiver.

See this issue for a diagram: https://github.com/openclimatefix/power_perceiver/issues/54
"""

# General imports
import datetime
import logging
import socket
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

import einops

# ML imports
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_msssim import ms_ssim
from torch import nn

from power_perceiver.analysis.log_national_pv import LogNationalPV
from power_perceiver.analysis.plot_probability_timeseries import LogProbabilityTimeseriesPlots
from power_perceiver.analysis.plot_satellite import LogSatellitePlots
from power_perceiver.analysis.plot_tsne import LogTSNEPlot

# power_perceiver imports
from power_perceiver.consts import (
    T0_IDX_30_MIN,
    X_OSGB_MEAN,
    X_OSGB_STD,
    Y_OSGB_MEAN,
    Y_OSGB_STD,
    BatchKey,
)
from power_perceiver.load_prepared_batches.data_sources.satellite import SAT_MEAN, SAT_STD
from power_perceiver.load_raw.data_sources.raw_gsp_data_source import RawGSPDataSource
from power_perceiver.load_raw.data_sources.raw_nwp_data_source import RawNWPDataSource
from power_perceiver.load_raw.data_sources.raw_pv_data_source import RawPVDataSource
from power_perceiver.load_raw.data_sources.raw_satellite_data_source import RawSatelliteDataSource
from power_perceiver.load_raw.national_pv_dataset import NationalPVDataset
from power_perceiver.load_raw.raw_dataset import RawDataset
from power_perceiver.np_batch_processor.encode_space_time import EncodeSpaceTime
from power_perceiver.np_batch_processor.save_t0_time import SaveT0Time
from power_perceiver.np_batch_processor.sun_position import SunPosition
from power_perceiver.np_batch_processor.topography import Topography
from power_perceiver.pytorch_modules.mixture_density_network import (
    MixtureDensityNetwork,
    get_distribution,
)
from power_perceiver.pytorch_modules.nwp_processor import NWPProcessor
from power_perceiver.pytorch_modules.query_generator import GSPQueryGenerator, PVQueryGenerator
from power_perceiver.pytorch_modules.satellite_predictor import XResUNet
from power_perceiver.pytorch_modules.satellite_processor import HRVSatelliteProcessor
from power_perceiver.pytorch_modules.self_attention import MultiLayerTransformerEncoder
from power_perceiver.transforms.pv import PVPowerRollingWindow
from power_perceiver.xr_batch_processor.reduce_num_timesteps import random_int_without_replacement

logging.basicConfig()
_log = logging.getLogger("power_perceiver")
_log.setLevel(logging.DEBUG)

plt.rcParams["figure.figsize"] = (18, 10)
plt.rcParams["figure.facecolor"] = "white"

# SatellitePredictor options
NUM_HIST_SAT_IMAGES = 12  # v15 pre-prepared batches use 7
NUM_FUTURE_SAT_IMAGES = 25  # v15 pre-prepared batches use 24
USE_TOPOGRAPHY = True
USE_SUN_POSITION = True

SATELLITE_PREDICTOR_IMAGE_WIDTH_PIXELS = 256
SATELLITE_PREDICTOR_IMAGE_HEIGHT_PIXELS = 128

SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS = 64
N_PV_SYSTEMS_PER_EXAMPLE = 8

# PowerPerceiver options
D_MODEL = 128
N_HEADS = 16


torch.manual_seed(42)

ON_DONATELLO = socket.gethostname() == "donatello"


def get_dataloader(
    start_date,
    end_date,
    num_workers,
    n_batches_per_epoch_per_worker,
    load_subset_every_epoch,
    train: bool,
) -> torch.utils.data.DataLoader:

    data_source_kwargs = dict(
        start_date=start_date,
        end_date=end_date,
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=2),
    )

    sat_data_source = RawSatelliteDataSource(
        zarr_path=(
            (
                "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
                "satellite/EUMETSAT/SEVIRI_RSS/zarr/v3/eumetsat_seviri_hrv_uk.zarr"
            )
            if ON_DONATELLO
            else (
                "gs://solar-pv-nowcasting-data/"
                "satellite/EUMETSAT/SEVIRI_RSS/v3/eumetsat_seviri_hrv_uk.zarr"
            )
        ),
        roi_height_pixels=SATELLITE_PREDICTOR_IMAGE_HEIGHT_PIXELS,
        roi_width_pixels=SATELLITE_PREDICTOR_IMAGE_WIDTH_PIXELS,
        **data_source_kwargs,
    )

    pv_data_source = RawPVDataSource(
        pv_power_filename="~/data/PV/Passiv/ocf_formatted/v0/passiv.netcdf",
        pv_metadata_filename="~/data/PV/Passiv/ocf_formatted/v0/system_metadata_OCF_ONLY.csv",
        roi_height_meters=96_000,
        roi_width_meters=96_000,
        n_pv_systems_per_example=N_PV_SYSTEMS_PER_EXAMPLE,
        transforms=[PVPowerRollingWindow(expect_dataset=False)],
        **data_source_kwargs,
    )

    gsp_data_source = RawGSPDataSource(
        gsp_pv_power_zarr_path="~/data/PV/GSP/v3/pv_gsp.zarr",
        gsp_id_to_region_id_filename="~/data/PV/GSP/eso_metadata.csv",
        sheffield_solar_region_path="~/data/PV/GSP/gsp_shape",
        **data_source_kwargs,
    )

    nwp_data_source = RawNWPDataSource(
        zarr_path=(
            (
                "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
                "NWP/UK_Met_Office/UKV/zarr/UKV_intermediate_version_3.zarr"
            )
            if ON_DONATELLO
            else ("gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/UKV_intermediate_version_3.zarr")
        ),
        roi_height_pixels=4,
        roi_width_pixels=4,
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=2),
        start_date=start_date,
        end_date=end_date,
        y_coarsen=16,
        x_coarsen=16,
        channels=["dswrf", "t", "si10", "prate"],
    )

    np_batch_processors = [
        EncodeSpaceTime(),
        SaveT0Time(pv_t0_idx=NUM_HIST_SAT_IMAGES - 1, gsp_t0_idx=T0_IDX_30_MIN),
    ]
    if USE_SUN_POSITION:
        np_batch_processors.append(SunPosition())
    if USE_TOPOGRAPHY:
        np_batch_processors.append(Topography("/home/jack/europe_dem_2km_osgb.tif"))

    raw_dataset_kwargs = dict(
        n_examples_per_batch=24,  # TODO: Increase to more like 32!
        n_batches_per_epoch=n_batches_per_epoch_per_worker,
        np_batch_processors=np_batch_processors,
        load_subset_every_epoch=load_subset_every_epoch,
        min_duration_to_load_per_epoch=(12 * 48) if ON_DONATELLO else (12 * 12),
    )

    if train:
        raw_dataset = RawDataset(
            data_source_combos=dict(
                sat_only=(sat_data_source,),
                gsp_pv_nwp_sat=(
                    gsp_data_source,
                    pv_data_source,
                    nwp_data_source,
                    deepcopy(sat_data_source),
                ),
            ),
            **raw_dataset_kwargs,
        )
    else:
        raw_dataset = NationalPVDataset(
            data_source_combos=dict(
                gsp_pv_nwp_sat=(gsp_data_source, pv_data_source, nwp_data_source, sat_data_source),
            ),
            **raw_dataset_kwargs,
        )

    if not num_workers:
        raw_dataset.per_worker_init(worker_id=0)

    def _worker_init_fn(worker_id: int):
        worker_info = torch.utils.data.get_worker_info()
        dataset_obj = worker_info.dataset
        dataset_obj.per_worker_init(worker_id=worker_id)

    dataloader = torch.utils.data.DataLoader(
        raw_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=True,
    )

    return dataloader


train_dataloader = get_dataloader(
    start_date="2020-01-01",
    end_date="2020-12-31",
    num_workers=2,
    n_batches_per_epoch_per_worker=2048,
    load_subset_every_epoch=True,
    train=True,
)

N_GSPS_AFTER_FILTERING = 313
val_dataloader = get_dataloader(
    start_date="2021-01-01",
    end_date="2021-12-31",
    num_workers=1,  # MUST BE 1! OTHERWISE LogNationalPV BREAKS!
    n_batches_per_epoch_per_worker=N_GSPS_AFTER_FILTERING,
    load_subset_every_epoch=False,
    train=False,
)


# ---------------------------------- SatellitePredictor ----------------------------------


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
class SatellitePredictor(pl.LightningModule):
    use_coord_conv: bool = False
    crop: bool = False
    use_topography: bool = USE_TOPOGRAPHY
    use_sun_position: bool = USE_SUN_POSITION

    # kwargs to fastai DynamicUnet. See this page for details:
    # https://fastai1.fast.ai/vision.models.unet.html#DynamicUnet
    pretrained: bool = False
    blur_final: bool = True  # Blur final layer. fastai default is True.
    self_attention: bool = True  # Use SA layer at the third block before the end.
    last_cross: bool = True  # Use a cross-connection with the direct input of the model.
    bottle: bool = False  # Bottleneck the last skip connection.

    def __post_init__(self):
        super().__init__()

        self.satellite_predictor = XResUNet(
            img_size=(
                SATELLITE_PREDICTOR_IMAGE_HEIGHT_PIXELS,
                SATELLITE_PREDICTOR_IMAGE_WIDTH_PIXELS,
            ),
            n_in=(
                NUM_HIST_SAT_IMAGES
                + (2 if self.use_coord_conv else 0)
                + (1 if self.use_topography else 0)
                + (2 if self.use_sun_position else 0)
            ),
            n_out=NUM_FUTURE_SAT_IMAGES,
            pretrained=self.pretrained,
            blur_final=self.blur_final,
            self_attention=self.self_attention,
            last_cross=self.last_cross,
            bottle=self.bottle,
        )

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        data = x[BatchKey.hrvsatellite_actual][
            :, :NUM_HIST_SAT_IMAGES, 0
        ]  # Shape: (example, time, y, x)
        height, width = data.shape[2:]
        assert height == SATELLITE_PREDICTOR_IMAGE_HEIGHT_PIXELS, f"{height=}"
        assert width == SATELLITE_PREDICTOR_IMAGE_WIDTH_PIXELS, f"{width=}"

        if self.use_coord_conv:
            osgb_coords = get_osgb_coords_for_coord_conv(x)
            data = torch.concat((data, osgb_coords), dim=1)

        if self.use_topography:
            surface_height = x[BatchKey.hrvsatellite_surface_height]
            surface_height = surface_height.unsqueeze(1)  # Add channel dim
            data = torch.concat((data, surface_height), dim=1)

        if self.use_sun_position:
            azimuth_at_t0 = x[BatchKey.hrvsatellite_solar_azimuth][:, NUM_HIST_SAT_IMAGES]
            elevation_at_t0 = x[BatchKey.hrvsatellite_solar_elevation][:, NUM_HIST_SAT_IMAGES]
            sun_pos = torch.stack((azimuth_at_t0, elevation_at_t0), dim=1)  # Shape: (example, 2)
            del azimuth_at_t0, elevation_at_t0
            # Repeat over y and x:
            sun_pos = einops.repeat(sun_pos, "example chan -> example chan y x", y=height, x=width)
            data = torch.concat((data, sun_pos), dim=1)

        assert data.isfinite().all()
        predicted_sat = self.satellite_predictor(data)
        assert predicted_sat.isfinite().all()
        return predicted_sat  # Shape: example, time, y, x


# ---------------------------------- SatelliteTransformer ----------------------------------


def maybe_pad_with_zeros(tensor: torch.Tensor, requested_dim: int) -> torch.Tensor:
    num_zeros_to_pad = requested_dim - tensor.shape[-1]
    assert num_zeros_to_pad >= 0, f"{requested_dim=}, {tensor.shape=}"
    if num_zeros_to_pad > 0:
        zero_padding_shape = tensor.shape[:2] + (num_zeros_to_pad,)
        zero_padding = torch.zeros(*zero_padding_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.concat((tensor, zero_padding), dim=2)
    return tensor


# Indexes for cropping the centre of the satellite image:
LEFT_IDX = (SATELLITE_PREDICTOR_IMAGE_WIDTH_PIXELS // 2) - (
    SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS // 2
)
RIGHT_IDX = LEFT_IDX + SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS
TOP_IDX = (SATELLITE_PREDICTOR_IMAGE_HEIGHT_PIXELS // 2) - (
    SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS // 2
)
BOTTOM_IDX = TOP_IDX + SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS


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
            num_gsps=NUM_GSPS,
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

        self.power_output = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
            nn.GELU(),
            nn.Linear(in_features=self.d_model, out_features=self.d_model // 2),
            nn.GELU(),
            nn.Linear(in_features=self.d_model // 2, out_features=1),
            nn.ReLU(),  # Ensure the output is always positive!
        )

    def forward(
        self, x: dict[BatchKey, torch.Tensor], hrvsatellite: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        # Reshape so each timestep is considered a different example:
        num_examples, num_timesteps = x[BatchKey.pv].shape[:2]
        # TODO: Fix this `original_x` hack, which is needed to prevent the reshaping affecting
        # the time_transformer, too!
        original_x = {}
        for batch_key in (
            # BatchKey.gsp_5_min_time_utc_fourier,
            BatchKey.pv_time_utc_fourier,
            BatchKey.hrvsatellite_solar_azimuth,
            BatchKey.hrvsatellite_solar_elevation,
            BatchKey.hrvsatellite_time_utc_fourier,
        ):
            original_x[batch_key] = x[batch_key]
            x[batch_key] = einops.rearrange(x[batch_key], "example time ... -> (example time) ...")

        hrvsatellite = einops.rearrange(hrvsatellite, "example time ... -> (example time) ...")

        # Process satellite data and queries:
        pv_query = self.pv_query_generator(x)
        gsp_query = self.gsp_query_generator(x, for_satellite_transformer=True)
        satellite_data = self.hrvsatellite_processor(x, hrvsatellite)

        # Pad with zeros if necessary to get up to self.d_model:
        pv_query = maybe_pad_with_zeros(pv_query, requested_dim=self.d_model)
        gsp_query = maybe_pad_with_zeros(gsp_query, requested_dim=self.d_model)
        satellite_data = maybe_pad_with_zeros(satellite_data, requested_dim=self.d_model)

        # Prepare inputs for the transformer_encoder:
        attn_input = torch.concat((pv_query, gsp_query, satellite_data), dim=1)

        # Mask the NaN GSP and PV queries. True or non-zero value indicates value will be ignored.
        mask = attn_input.isnan().any(dim=2)
        attn_input = attn_input.nan_to_num(0)

        # Pass data into transformer_encoder:
        attn_output = attn_input + self.transformer_encoder(attn_input, src_key_padding_mask=mask)

        # Reshape to (example time element d_model):
        attn_output = einops.rearrange(
            attn_output,
            "(example time) ... -> example time ...",
            example=num_examples,
            time=num_timesteps,
        )

        assert attn_output.isfinite().all()

        # Select the elements of the output which correspond to the query:
        gsp_start_idx = pv_query.shape[1]
        gsp_end_idx = gsp_start_idx + gsp_query.shape[1]
        pv_attn_out = attn_output[:, :, :gsp_start_idx]
        gsp_attn_out = attn_output[:, :, gsp_start_idx:gsp_end_idx]

        # Power output:
        pv_power_out = self.power_output(pv_attn_out).squeeze()
        gsp_power_out = self.power_output(gsp_attn_out).squeeze()

        # Put back the original data! TODO: Remove this hack!
        x.update(original_x)

        return {
            "pv_attn_out": pv_attn_out,  # shape: (example, 5_min_time, n_pv_systems, d_model)
            "gsp_attn_out": gsp_attn_out,  # shape: (example, 5_min_time, 1, d_model)
            "pv_power_out": pv_power_out,  # shape: (example, 5_min_time, n_pv_systems)
            "gsp_power_out": gsp_power_out,  # shape: (example, 5_min_time)
        }


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class FullModel(pl.LightningModule):
    d_model: int = D_MODEL
    pv_system_id_embedding_dim: int = 16
    num_heads: int = N_HEADS
    dropout: float = 0.1
    share_weights_across_latent_transformer_layers: bool = False
    num_latent_transformer_encoders: int = 4
    cheat: bool = False  #: Use real satellite imagery of the future.
    stop_gradients_before_unet: bool = False
    #: Compute the loss on a central crop of the imagery.
    crop_sat_before_sat_predictor_loss: bool = False
    num_5_min_history_timesteps_during_training: Optional[int] = 4
    num_5_min_forecast_timesteps_during_training: Optional[int] = 6
    num_gaussians: int = 2
    num_rnn_layers: int = 4

    def __post_init__(self):
        super().__init__()

        if self.cheat:
            _log.warning("CHEATING MODE ENABLED! Using real satellite imagery of future!")

        # Predict future satellite images:
        self.satellite_predictor = SatellitePredictor()

        # Load SatellitePredictor weights
        self.satellite_predictor.load_state_dict(
            torch.load(
                (
                    "/home/jack/dev/ocf/power_perceiver/experiments/power_perceiver/3qvkf1dy/"
                    "checkpoints/epoch=170-step=175104-just-satellite-predictor.state_dict.pth"
                )
                if ON_DONATELLO
                else (
                    "/home/jack/model_params/satellite_predictor/"
                    "epoch=170-step=175104-just-satellite-predictor.state_dict.pth"
                )
            )
        )

        # Infer GSP and PV power output for a single timestep of satellite imagery.
        self.satellite_transformer = SatelliteTransformer()

        self.nwp_processor = NWPProcessor()

        # Find temporal features, and help calibrate predictions using recent history:
        # TODO: Move the RNN into its own nn.Module.
        rnn_kwargs = dict(
            hidden_size=self.d_model,
            num_layers=self.num_rnn_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.pv_rnn_history_encoder = nn.RNN(
            # Each timestep of the encoder RNN receives the output of the satellite_transformer
            # (which is of size d_model per PV system and per timestep), for a single PV system,
            # plus one timestep of that PV system's history.
            input_size=self.d_model + 1,
            **rnn_kwargs,
        )
        self.pv_rnn_future_decoder = nn.RNN(
            # Each timestep of the decoder RNN receives the output of the satellite_transformer
            # (which is of size d_model per PV system and per timestep) for a single PV system.
            input_size=self.d_model,
            **rnn_kwargs,
        )

        # Look across timesteps to produce the final output.
        self.time_transformer = MultiLayerTransformerEncoder(
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout,
            share_weights_across_latent_transformer_layers=(
                self.share_weights_across_latent_transformer_layers
            ),
            num_latent_transformer_encoders=self.num_latent_transformer_encoders,
        )

        self.pv_mixture_density_net = nn.Sequential(
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
            nn.GELU(),
            nn.Linear(in_features=self.d_model, out_features=self.d_model),
            nn.GELU(),
            MixtureDensityNetwork(in_features=self.d_model, num_gaussians=self.num_gaussians),
        )

        # Do this at the end of __post_init__ to capture model topology to wandb:
        self.save_hyperparameters()

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Predict future satellite images. The SatellitePredictor always gets every timestep.
        if self.cheat:
            predicted_sat = x[BatchKey.hrvsatellite_actual][:, :NUM_HIST_SAT_IMAGES, 0]
        else:
            predicted_sat = self.satellite_predictor(x=x)  # Shape: example, time, y, x

        if self.stop_gradients_before_unet:
            predicted_sat = predicted_sat.detach()

        hrvsatellite = torch.concat(
            (x[BatchKey.hrvsatellite_actual][:, :NUM_HIST_SAT_IMAGES, 0], predicted_sat), dim=1
        )
        assert hrvsatellite.isfinite().all()

        # Select a subset of 5-minute timesteps during training:
        if self.num_5_min_history_timesteps_during_training and self.training:
            random_history_timestep_indexes = random_int_without_replacement(
                start=0,
                stop=NUM_HIST_SAT_IMAGES,
                num=self.num_5_min_history_timesteps_during_training,
            )
            random_forecast_timestep_indexes = random_int_without_replacement(
                start=NUM_HIST_SAT_IMAGES,
                stop=NUM_HIST_SAT_IMAGES + NUM_FUTURE_SAT_IMAGES,
                num=self.num_5_min_forecast_timesteps_during_training,
            )
            random_timestep_indexes = np.concatenate(
                (random_history_timestep_indexes, random_forecast_timestep_indexes)
            )
            hrvsatellite = hrvsatellite[:, random_timestep_indexes]
            for batch_key in (
                # Don't include BatchKey.hrvsatellite, because we need all timesteps to
                # compute the loss for the SatellitePredictor!
                # Don't include BatchKey.hrvsatellite_time_utc because all it's used for
                # is plotting satellite predictions, and we need all timesteps for that!
                # Don't subselect pv_time_utc here, because we subselect that in
                # `plot_probabability_timeseries.plot_pv_power`.
                # We *do* subset hrvsatellite_time_utc_fourier because it's used in the
                # satellite_transformer.
                BatchKey.hrvsatellite_time_utc_fourier,
                BatchKey.pv,
                BatchKey.pv_time_utc_fourier,
                BatchKey.hrvsatellite_solar_azimuth,
                BatchKey.hrvsatellite_solar_elevation,
            ):
                x[batch_key] = x[batch_key][:, random_timestep_indexes]
            num_5_min_timesteps = (
                self.num_5_min_history_timesteps_during_training
                + self.num_5_min_forecast_timesteps_during_training
            )
        else:
            num_5_min_timesteps = NUM_HIST_SAT_IMAGES + NUM_FUTURE_SAT_IMAGES
            random_timestep_indexes = None

        # Detach because it looks like it hurts performance to let the gradients go backwards
        # from here
        hrvsatellite = hrvsatellite.detach()

        # Crop satellite data:
        # This is necessary because we want to give the "satellite predictor"
        # a large rectangle of imagery (e.g. 256 wide x 128 high) so it can see clouds
        # coming, but we probably don't want to give that huge image to a full-self-attention model.
        hrvsatellite = hrvsatellite[..., TOP_IDX:BOTTOM_IDX, LEFT_IDX:RIGHT_IDX]
        assert hrvsatellite.shape[-2] == SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS
        assert hrvsatellite.shape[-1] == SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS

        # Crop spatial coordinates:
        original_x = {}
        for batch_key in (
            BatchKey.hrvsatellite_y_osgb_fourier,
            BatchKey.hrvsatellite_x_osgb_fourier,
            BatchKey.hrvsatellite_surface_height,
        ):
            # Save a backup so we can put the backup back at the end of this function!
            # This is useful so we can plot the full surface height.
            original_x[batch_key] = x[batch_key]
            x[batch_key] = x[batch_key][:, TOP_IDX:BOTTOM_IDX, LEFT_IDX:RIGHT_IDX]
            assert x[batch_key].shape[1] == SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS
            assert x[batch_key].shape[2] == SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS

        # Pass through the transformer!
        assert hrvsatellite.isfinite().all()
        sat_trans_out = self.satellite_transformer(x=x, hrvsatellite=hrvsatellite)
        sat_trans_pv_attn_out = sat_trans_out[
            "pv_attn_out"
        ]  # Shape: (example time n_pv_systems d_model)
        sat_trans_gsp_attn_out = sat_trans_out["gsp_attn_out"]

        assert sat_trans_pv_attn_out.isfinite().all()
        assert sat_trans_gsp_attn_out.isfinite().all()

        # Prepare inputs for the pv_rnn_history_encoder:
        # Each timestep of the encoder RNN receives the output of the satellite_transformer
        # (which is of size d_model per PV system and per timestep), for a single PV system,
        # plus one timestep of that PV system's history.
        # Some of `x[BatchKey.pv]` will be NaN. But that's OK. We mask missing examples.
        # And use nan_to_num later in this function.
        hist_pv = x[BatchKey.pv][:, : self.t0_idx_5_min + 1]  # Shape: example, time, n_pv_systems

        # Reshape so each PV system is seen as a different example:
        hist_pv = einops.rearrange(
            hist_pv, "example time n_pv_systems -> (example n_pv_systems) time 1"
        )
        sat_trans_pv_attn_out = einops.rearrange(
            sat_trans_pv_attn_out,
            "example time n_pv_systems d_model -> (example n_pv_systems) time d_model",
            n_pv_systems=N_PV_SYSTEMS_PER_EXAMPLE,  # sanity check
            time=num_5_min_timesteps,
        )
        hist_sat_trans_pv_attn_out = sat_trans_pv_attn_out[:, : self.t0_idx_5_min + 1]

        pv_rnn_hist_enc_in = torch.concat((hist_pv, hist_sat_trans_pv_attn_out), dim=2)
        pv_rnn_hist_enc_in = pv_rnn_hist_enc_in.nan_to_num(0)
        pv_rnn_hist_enc_out, pv_rnn_hist_enc_hidden = self.pv_rnn_history_encoder(
            pv_rnn_hist_enc_in
        )

        # Now for the pv_rnn_future_decoder:
        future_sat_trans_pv_attn_out = sat_trans_pv_attn_out[:, self.t0_idx_5_min + 1 :]
        future_sat_trans_pv_attn_out = future_sat_trans_pv_attn_out.nan_to_num(0)
        pv_rnn_fut_dec_out, _ = self.pv_rnn_future_decoder(
            future_sat_trans_pv_attn_out,
            pv_rnn_hist_enc_hidden,
        )

        # Concatenate the output from the encoder and decoder RNNs, and reshape
        # so each timestep and each PV system is a separate element into `time_transformer`.
        pv_rnn_out = torch.concat((pv_rnn_hist_enc_out, pv_rnn_fut_dec_out), dim=1)
        # rnn_out shape: (example n_pv_systems), time, d_model
        assert pv_rnn_out.isfinite().all()
        pv_rnn_out = einops.rearrange(
            pv_rnn_out,
            "(example n_pv_systems) time d_model -> example (time n_pv_systems) d_model",
            n_pv_systems=N_PV_SYSTEMS_PER_EXAMPLE,
            d_model=self.d_model,
            time=num_5_min_timesteps,
        )
        n_pv_elements = pv_rnn_out.shape[1]
        del sat_trans_pv_attn_out, hist_sat_trans_pv_attn_out, future_sat_trans_pv_attn_out
        del hist_pv, pv_rnn_hist_enc_hidden

        # Reshape gsp attention outputs so each timestep is
        # seen as a separate element into the `time transformer`. Remember that, at this stage,
        # the sat_trans_gsp_attn_out is 5-minutely.
        sat_trans_gsp_attn_out = einops.rearrange(
            sat_trans_gsp_attn_out, "example time 1 d_model -> example time d_model"
        )

        # Pad with zeros
        pv_rnn_out = maybe_pad_with_zeros(pv_rnn_out, requested_dim=self.d_model)
        sat_trans_gsp_attn_out = maybe_pad_with_zeros(
            sat_trans_gsp_attn_out, requested_dim=self.d_model
        )

        # Get GSP query for the time_transformer:
        # The query for the time_transformer is just for the half-hourly timesteps
        # (not the half-hourly timesteps resampled to 5-minutely.)
        gsp_query_generator = self.satellite_transformer.gsp_query_generator
        gsp_query = gsp_query_generator(x, for_satellite_transformer=False)
        gsp_query = maybe_pad_with_zeros(gsp_query, requested_dim=self.d_model)

        # Prepare NWP inputs
        nwp_query = self.nwp_processor(x)
        nwp_query = maybe_pad_with_zeros(nwp_query, requested_dim=self.d_model)

        # Concatenate all the things we're going to feed into the "time transformer":
        # `pv_rnn_out` must be the first set of elements.
        # `gsp_query` must be the last set of elements.
        time_attn_in = torch.concat(
            (pv_rnn_out, sat_trans_gsp_attn_out, nwp_query, gsp_query), dim=1
        )

        # Some whole examples don't include GSP (or PV) data.
        # Here, we set those to zero so the model trains without NaN loss, and mask them.
        # Examples with NaN PV or GSP power are masked in the objective function and
        # in the attention mechanism.
        # TODO: Maybe we shouldn't actually be masking entire *examples*?
        # Masking entire examples seems to break the downstream code.
        # Maybe it's sufficient to mask the loss?
        # If so, we can remove the block of code below :) See issue #103.
        # mask = torch.concat(
        #     (
        #         einops.rearrange(
        #             x[BatchKey.pv].isnan(),
        #             "example time n_pv_systems -> example (time n_pv_systems)",
        #         ),
        #         # Remember that `sat_trans_gsp_attn_out` is at *5 minute* intervals!
        #         # So we need to repeat the GSP mask for each 5 minute interval.
        #         # And the might have less 5 minute intervals if we're training and
        #         # subsampling 5-min intervals!
        #         einops.repeat(
        #             x[BatchKey.gsp_id].isnan().squeeze(),
        #             "example -> example num_5_min_timesteps",
        #             num_5_min_timesteps=num_5_min_timesteps,
        #         ),
        #         gsp_query.isnan().any(dim=2),
        #     ),
        #     dim=1,
        # )
        # assert not mask.all()
        # self.log(f"{self.tag}/time_transformer_mask_mean", mask.float().mean())
        time_attn_in = time_attn_in.nan_to_num(0)
        time_attn_out = self.time_transformer(
            time_attn_in
        )  # TODO Put mask back in?, src_key_padding_mask=mask)
        # If we're using `mask` then time_attn_out will be NaN for examples which
        # are entirely masked (because this example has no PV or GSP)

        # The MDN doesn't like NaNs:
        time_attn_out = time_attn_out.nan_to_num(0)

        # Reshape the PV power predictions
        predicted_pv_power = self.pv_mixture_density_net(time_attn_out[:, :n_pv_elements])
        predicted_pv_power = einops.rearrange(
            predicted_pv_power,
            "example (time n_pv_systems) mdn_features -> example time n_pv_systems mdn_features",
            time=num_5_min_timesteps,
            n_pv_systems=N_PV_SYSTEMS_PER_EXAMPLE,
            mdn_features=self.num_gaussians * 3,  # x3 for pi, mu, sigma.
        )

        # GSP power. There's just 1 GSP. So each gsp element is a timestep.
        n_gsp_elements = gsp_query.shape[1]
        predicted_gsp_power = self.pv_mixture_density_net(time_attn_out[:, -n_gsp_elements:])

        x.update(original_x)

        return dict(
            predicted_pv_power=predicted_pv_power,  # Shape: (example time n_pv_sys mdn_features)
            predicted_gsp_power=predicted_gsp_power,  # Shape: (example time mdn_features)
            predicted_sat=predicted_sat,  # Shape: example, time, y, x
            # shape: (example, 5_min_time, n_pv_systems):
            pv_power_from_sat_transformer=sat_trans_out["pv_power_out"],
            # shape: (example, 5_min_time):
            gsp_power_from_sat_transformer=sat_trans_out["gsp_power_out"],
            random_timestep_indexes=random_timestep_indexes,
        )

    @property
    def t0_idx_5_min(self) -> int:
        if self.training and self.num_5_min_history_timesteps_during_training:
            return self.num_5_min_history_timesteps_during_training - 1
        else:
            return NUM_HIST_SAT_IMAGES - 1

    @property
    def tag(self) -> str:
        return "train" if self.training else "validation"

    def training_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        return self.validation_step(batch, batch_idx)

    def validation_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:

        network_out = self(batch)

        # PV & GSP LOSS ######################
        predicted_pv_power = network_out["predicted_pv_power"]
        actual_pv_power = batch[BatchKey.pv]
        predicted_gsp_power = network_out["predicted_gsp_power"]
        actual_gsp_power = batch[BatchKey.gsp].squeeze()

        # Mask predicted and actual PV and GSP (some examples don't have PV and/or GSP)
        # For more discussion of how to mask losses in pytorch, see:
        # https://discuss.pytorch.org/t/masking-input-to-loss-function/121830/3
        pv_mask = actual_pv_power.isfinite()
        pv_mask_from_t0 = pv_mask[:, self.t0_idx_5_min + 1 :]
        gsp_mask = actual_gsp_power.isfinite()
        gsp_mask_from_t0 = gsp_mask[:, T0_IDX_30_MIN + 1 :]

        predicted_pv_power_from_t0 = predicted_pv_power[:, self.t0_idx_5_min + 1 :][pv_mask_from_t0]
        actual_pv_power_from_t0 = actual_pv_power[:, self.t0_idx_5_min + 1 :][pv_mask_from_t0]

        predicted_gsp_power_from_t0 = predicted_gsp_power[:, T0_IDX_30_MIN + 1 :][gsp_mask_from_t0]
        actual_gsp_power_from_t0 = actual_gsp_power[:, T0_IDX_30_MIN + 1 :][gsp_mask_from_t0]

        # PV negative log prob loss:
        pv_distribution = get_distribution(predicted_pv_power_from_t0)
        pv_neg_log_prob_loss = -pv_distribution.log_prob(actual_pv_power_from_t0).mean()
        self.log(f"{self.tag}/pv_neg_log_prob", pv_neg_log_prob_loss)

        # PV power loss:
        pv_mse_loss = F.mse_loss(pv_distribution.mean, actual_pv_power_from_t0)
        self.log(f"{self.tag}/pv_mse", pv_mse_loss)

        pv_nmae_loss = F.l1_loss(pv_distribution.mean, actual_pv_power_from_t0)
        self.log(f"{self.tag}/pv_nmae", pv_nmae_loss)

        # PV loss from satellite transformer:
        # (don't do this for GSP because GSP data from sat transformer is 5-minutely!)
        # We include the history as well as the forecast, because the satellite transformer
        # doesn't see the history, and we want to encourage the satellite transformer to
        # infer PV yield from historical and future images.
        pv_from_sat_trans_mse_loss = F.mse_loss(
            network_out["pv_power_from_sat_transformer"][pv_mask], actual_pv_power[pv_mask]
        )
        self.log(f"{self.tag}/pv_from_sat_trans_mse_loss", pv_from_sat_trans_mse_loss)

        # GSP negative log prob loss:
        gsp_distribution_masked = get_distribution(predicted_gsp_power_from_t0)
        gsp_neg_log_prob_loss = -gsp_distribution_masked.log_prob(actual_gsp_power_from_t0).mean()
        self.log(f"{self.tag}/gsp_neg_log_prob", gsp_neg_log_prob_loss)

        # GSP power loss:
        gsp_mse_loss = F.mse_loss(gsp_distribution_masked.mean, actual_gsp_power_from_t0)
        self.log(f"{self.tag}/gsp_mse", gsp_mse_loss)

        gsp_nmae_loss = F.l1_loss(gsp_distribution_masked.mean, actual_gsp_power_from_t0)
        self.log(f"{self.tag}/gsp_nmae", gsp_nmae_loss)

        # Total PV and GSP loss:
        total_pv_and_gsp_mse_loss = gsp_mse_loss + pv_mse_loss + pv_from_sat_trans_mse_loss
        total_pv_and_gsp_neg_log_prob_loss = gsp_neg_log_prob_loss + pv_neg_log_prob_loss
        self.log(f"{self.tag}/total_mse", total_pv_and_gsp_mse_loss)

        # Total NMAE loss:
        total_nmae_loss = pv_nmae_loss + gsp_nmae_loss
        self.log(f"{self.tag}/total_nmae", total_nmae_loss)

        # SATELLITE PREDICTOR LOSS ################
        predicted_sat = network_out["predicted_sat"]
        actual_sat = batch[BatchKey.hrvsatellite_actual][:, NUM_HIST_SAT_IMAGES:, 0]

        sat_mse_loss = F.mse_loss(predicted_sat, actual_sat)
        self.log(f"{self.tag}/sat_mse", sat_mse_loss)

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
        self.log(f"{self.tag}/ms_ssim", ms_ssim_loss)
        self.log(f"{self.tag}/ms_ssim+sat_mse", ms_ssim_loss + sat_mse_loss)

        if self.crop_sat_before_sat_predictor_loss:
            # Loss on a central crop:
            # The cropped image has to be larger than 32x32 otherwise ms-ssim complains:
            # "Image size should be larger than 32 due to the 4 downsamplings in ms-ssim"
            sat_mse_loss_crop = F.mse_loss(
                predicted_sat[:, :, TOP_IDX:BOTTOM_IDX, LEFT_IDX:RIGHT_IDX],
                actual_sat[:, :, TOP_IDX:BOTTOM_IDX, LEFT_IDX:RIGHT_IDX],
            )
            self.log(f"{self.tag}/sat_mse_crop", sat_mse_loss_crop)
            ms_ssim_loss_crop = 1 - ms_ssim(
                predicted_sat_denorm[:, :, TOP_IDX:BOTTOM_IDX, LEFT_IDX:RIGHT_IDX],
                actual_sat_denorm[:, :, TOP_IDX:BOTTOM_IDX, LEFT_IDX:RIGHT_IDX],
                data_range=1023.0,
                size_average=True,  # Return a scalar.
                win_size=3,  # The Illinois ClimateHack folks used win_size=3.
            )
            self.log(f"{self.tag}/ms_ssim_crop", ms_ssim_loss_crop)
            self.log(f"{self.tag}/ms_ssim_crop+sat_mse_crop", ms_ssim_loss_crop + sat_mse_loss_crop)
            sat_loss = ms_ssim_loss_crop + sat_mse_loss_crop
        else:
            sat_loss = ms_ssim_loss + sat_mse_loss

        total_sat_pv_gsp_loss = sat_loss + total_pv_and_gsp_mse_loss
        total_sat_and_pv_gsp_neg_log_prob = sat_loss + total_pv_and_gsp_neg_log_prob_loss
        self.log(f"{self.tag}/total_sat_pv_gsp_loss", total_sat_pv_gsp_loss)
        self.log(f"{self.tag}/total_sat_and_pv_gsp_neg_log_prob", total_sat_and_pv_gsp_neg_log_prob)
        total_sat_and_pv_gsp_neg_log_prob_and_sat_trans_mse = (
            total_sat_and_pv_gsp_neg_log_prob + pv_from_sat_trans_mse_loss
        )
        self.log(
            f"{self.tag}/total_sat_and_pv_gsp_neg_log_prob_and_sat_trans_mse",
            total_sat_and_pv_gsp_neg_log_prob_and_sat_trans_mse,
        )

        return {
            "loss": total_sat_and_pv_gsp_neg_log_prob_and_sat_trans_mse,
            "predicted_gsp_power": predicted_gsp_power,
            "predicted_gsp_power_mean": get_distribution(predicted_gsp_power).mean,
            "actual_gsp_power": actual_gsp_power,
            "gsp_time_utc": batch[BatchKey.gsp_time_utc],
            "actual_pv_power": actual_pv_power,
            "predicted_pv_power": predicted_pv_power,
            "predicted_pv_power_mean": get_distribution(predicted_pv_power).mean,
            "predicted_sat": predicted_sat,  # Shape: example, time, y, x
            "actual_sat": actual_sat,
            "pv_power_from_sat_transformer": network_out["pv_power_from_sat_transformer"],
            "gsp_power_from_sat_transformer": network_out["gsp_power_from_sat_transformer"],
            "random_timestep_indexes": network_out["random_timestep_indexes"],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

        def _lr_lambda(epoch):
            return 50 / (epoch + 50)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, verbose=True)
        return [optimizer], [scheduler]


# ---------------------------------- Training ----------------------------------

model = FullModel()

wandb_logger = WandbLogger(
    name=(
        "025.08: Hist GSP. Use pretrained SatPred weights but detach. NWPs."
        " SatTrans in obj function. RNN for PV. GCP-1 with dual GPU."
    ),
    project="power_perceiver",
    entity="openclimatefix",
    log_model=True,
)


if ON_DONATELLO:
    GPUS = [0, 2]
else:  # On GCP
    GPUS = [0, 1]

# WARNING: Don't run multiple GPUs in ipython.
trainer = pl.Trainer(
    gpus=GPUS,
    strategy="ddp" if len(GPUS) > 1 else None,
    max_epochs=200,
    logger=wandb_logger,
    callbacks=[
        # Save the top 3 model params
        pl.callbacks.ModelCheckpoint(
            monitor="validation/gsp_nmae",
            mode="min",
            save_top_k=3,
        ),
        # Always save the most recent model (so we can resume training)
        pl.callbacks.ModelCheckpoint(filename="{epoch}"),
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        LogProbabilityTimeseriesPlots(),
        LogTSNEPlot(query_generator_name="satellite_transformer.pv_query_generator"),
        LogSatellitePlots(),
        LogNationalPV(),
    ],
)

trainer.fit(
    model=model,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
