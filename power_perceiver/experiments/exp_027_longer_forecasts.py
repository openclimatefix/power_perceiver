"""
U-Net followed by the Power Perceiver.

See this issue for a diagram: https://github.com/openclimatefix/power_perceiver/issues/54
"""

# General imports
import datetime
import logging
import socket
from dataclasses import dataclass
from typing import Optional

import einops

# ML imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import WandbLogger
from pytorch_msssim import ms_ssim
from torch import nn

from power_perceiver.analysis.log_national_pv import LogNationalPV
from power_perceiver.analysis.plot_probability_timeseries import LogProbabilityTimeseriesPlots
from power_perceiver.analysis.plot_tsne import LogTSNEPlot

# power_perceiver imports
from power_perceiver.consts import X_OSGB_MEAN, X_OSGB_STD, Y_OSGB_MEAN, Y_OSGB_STD, BatchKey
from power_perceiver.load_prepared_batches.data_sources.satellite import SAT_MEAN, SAT_STD
from power_perceiver.load_raw.data_sources.raw_gsp_data_source import RawGSPDataSource
from power_perceiver.load_raw.data_sources.raw_nwp_data_source import RawNWPDataSource
from power_perceiver.load_raw.data_sources.raw_pv_data_source import RawPVDataSource
from power_perceiver.load_raw.data_sources.raw_satellite_data_source import RawSatelliteDataSource
from power_perceiver.load_raw.national_pv_dataset import NationalPVDataset
from power_perceiver.load_raw.raw_dataset import RawDataset
from power_perceiver.np_batch_processor.align_gsp_to_5_min import AlignGSPTo5Min
from power_perceiver.np_batch_processor.delete_forecast_satellite_imagery import (
    DeleteForecastSatelliteImagery,
)
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
from power_perceiver.utils import assert_num_dims, pandas_periods_to_our_periods_dt
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
D_MODEL = 256  # Must be divisible by N_HEADS
N_HEADS = 32

NWP_CHANNELS = ("t", "dswrf", "prate", "r", "si10", "vis", "lcc", "mcc", "hcc")

ON_DONATELLO = socket.gethostname() == "donatello"

DEBUG = False
ENABLE_WANDB = True

if DEBUG:
    GPUS = [0]
elif ON_DONATELLO:
    GPUS = [0, 1, 2, 4]
else:  # On GCP
    GPUS = [0, 1]


def get_dataloader(
    time_periods,
    num_workers,
    n_batches_per_epoch_per_worker,
    load_subset_every_epoch,
    train: bool,
) -> torch.utils.data.DataLoader:

    data_source_kwargs = dict(
        time_periods=time_periods,
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
        pv_power_filename="/home/jack/data/PV/Passiv/ocf_formatted/v0/passiv.netcdf",
        pv_metadata_filename="/home/jack/data/PV/Passiv/ocf_formatted/v0/system_metadata_OCF_ONLY.csv",
        roi_height_meters=96_000,
        roi_width_meters=96_000,
        n_pv_systems_per_example=N_PV_SYSTEMS_PER_EXAMPLE,
        transforms=[PVPowerRollingWindow(expect_dataset=False)],
        **data_source_kwargs,
    )

    gsp_data_source = RawGSPDataSource(
        gsp_pv_power_zarr_path="/home/jack/data/PV/GSP/v3/pv_gsp.zarr",
        gsp_id_to_region_id_filename="/home/jack/data/PV/GSP/eso_metadata.csv",
        sheffield_solar_region_path="/home/jack/data/PV/GSP/gsp_shape",
        time_periods=time_periods,
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=8),
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
        y_coarsen=16,
        x_coarsen=16,
        channels=NWP_CHANNELS,
        time_periods=time_periods,
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=8),
    )

    np_batch_processors = [AlignGSPTo5Min(), EncodeSpaceTime(), SaveT0Time()]
    if USE_SUN_POSITION:
        for modality_name in ["hrvsatellite", "gsp", "gsp_5_min", "pv", "nwp_target_time"]:
            np_batch_processors.append(SunPosition(modality_name=modality_name))
    if USE_TOPOGRAPHY:
        np_batch_processors.append(Topography("/home/jack/europe_dem_2km_osgb.tif"))
    # Delete imagery of the future, because we're not training the U-Net,
    # and we want to save GPU RAM.
    # But we do want hrvsatellite_time_utc to continue into the future by 2 hours because
    # downstream code relies on hrvsatellite_time_utc.
    # This must come last.
    np_batch_processors.append(
        DeleteForecastSatelliteImagery(num_hist_sat_images=NUM_HIST_SAT_IMAGES)
    )

    raw_dataset_kwargs = dict(
        n_examples_per_batch=14,  # TODO: Increase to more like 32!
        n_batches_per_epoch=n_batches_per_epoch_per_worker,
        np_batch_processors=np_batch_processors,
        load_subset_every_epoch=load_subset_every_epoch,
        min_duration_to_load_per_epoch=datetime.timedelta(
            hours=48 if DEBUG else ((12 * 32) if ON_DONATELLO else (12 * 24))
        ),
        data_source_combos=dict(
            gsp_pv_nwp_sat=(gsp_data_source, pv_data_source, nwp_data_source, sat_data_source),
        ),
    )

    if train:
        raw_dataset = RawDataset(**raw_dataset_kwargs)
    else:
        raw_dataset = NationalPVDataset(**raw_dataset_kwargs)

    if not num_workers:
        raw_dataset.per_worker_init(worker_id=0)

    def _worker_init_fn(worker_id: int):
        worker_info = torch.utils.data.get_worker_info()
        dataset_obj = worker_info.dataset
        if len(GPUS) > 1:
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        _log.info(f"{num_workers=} {worker_id=} {rank=} {worker_info.seed=}")
        # We set the worker_id to be unique across all GPUs, so each worker
        # sets a unique (but repeatable) random number generator seed.
        # We times by 16 just to really make sure that the random seeds
        # are different for each GPU.
        seed = worker_info.seed + worker_id + (rank * num_workers * 16)
        dataset_obj.per_worker_init(worker_id=worker_id, seed=seed)

    dataloader = torch.utils.data.DataLoader(
        raw_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
    )

    return dataloader


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

    def validation_step(
        self, batch: dict[BatchKey, torch.Tensor], batch_idx: int
    ) -> dict[str, object]:
        predicted_sat = self(batch)
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

        return dict(
            lost=sat_loss,
        )


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


def _crop_satellite_and_spatial_coords(
    x: dict[BatchKey, torch.Tensor]
) -> dict[BatchKey, torch.Tensor]:
    """Crop satellite data & spatial coordinates.

    This is necessary because we want to give the "satellite predictor"
    a large rectangle of imagery (e.g. 256 wide x 128 high) so it can see clouds
    coming, but we probably don't want to give that huge image to a full-self-attention model.
    """
    assert_num_dims(x[BatchKey.hrvsatellite_actual], num_expected_dims=5)
    assert_num_dims(x[BatchKey.hrvsatellite_predicted], num_expected_dims=4)

    def _check_shape(batch_key, y_idx=1, x_idx=2):
        error_msg = f"{batch_key.name}.shape = {x[batch_key].shape}"
        assert x[batch_key].shape[y_idx] == SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS, error_msg
        assert x[batch_key].shape[x_idx] == SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS, error_msg

    # Crop the predicted and actual imagery:
    for batch_key in (BatchKey.hrvsatellite_actual, BatchKey.hrvsatellite_predicted):
        x[batch_key] = x[batch_key][..., TOP_IDX:BOTTOM_IDX, LEFT_IDX:RIGHT_IDX]
        _check_shape(batch_key, y_idx=-2, x_idx=-1)

    # Crop the coords (which have to be done in a separate loop, because `y` and `x`
    # are at different positions):
    for batch_key in (
        BatchKey.hrvsatellite_y_osgb_fourier,
        BatchKey.hrvsatellite_x_osgb_fourier,
        BatchKey.hrvsatellite_surface_height,
    ):
        x[batch_key] = x[batch_key][:, TOP_IDX:BOTTOM_IDX, LEFT_IDX:RIGHT_IDX]
        _check_shape(batch_key)
    return x


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
    num_latent_transformer_encoders: int = 8

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

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Reshape so each timestep is considered a different example:
        num_examples, num_timesteps = x[BatchKey.pv].shape[:2]

        # Process satellite data and queries:
        pv_query = self.pv_query_generator(x)  # shape: example * time, n_pv_systems, features
        gsp_query = self.gsp_query_generator(
            x,
            include_history=True,
            base_batch_key=BatchKey.gsp_5_min,
            do_reshape_time_as_batch=True,
        )  # shape: example * time, 1, features
        satellite_data = self.hrvsatellite_processor(x)  # shape: example * time, positions, feats.

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

        return {
            "pv_attn_out": pv_attn_out,  # shape: (example, 5_min_time, n_pv_systems, d_model)
            "gsp_attn_out": gsp_attn_out,  # shape: (example, 5_min_time, 1, d_model)
        }


@dataclass(eq=False)
class PVRNN(nn.Module):
    hidden_size: int
    num_layers: int
    dropout: float

    def __post_init__(self):
        super().__init__()

        rnn_kwargs = dict(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.pv_rnn_history_encoder = nn.RNN(
            # Each timestep of the encoder RNN receives the output of the satellite_transformer
            # (which is of size d_model per PV system and per timestep), for a single PV system,
            # plus one timestep of that PV system's history.
            input_size=self.hidden_size + 1,
            **rnn_kwargs,
        )
        self.pv_rnn_future_decoder = nn.RNN(
            # Each timestep of the decoder RNN receives the output of the satellite_transformer
            # (which is of size d_model per PV system and per timestep) for a single PV system.
            input_size=self.hidden_size,
            **rnn_kwargs,
        )

    def forward(
        self, x: dict[BatchKey, torch.Tensor], sat_trans_pv_attn_out: torch.Tensor
    ) -> torch.Tensor:
        # Prepare inputs for the pv_rnn_history_encoder:
        # Each timestep of the encoder RNN receives the output of the satellite_transformer
        # (which is of size d_model per PV system and per timestep), for a single PV system,
        # plus one timestep of that PV system's history.
        # Some of `x[BatchKey.pv]` will be NaN. But that's OK. We mask missing examples.
        # And use nan_to_num later in this function.
        pv_t0_idx = x[BatchKey.pv_t0_idx]
        num_pv_timesteps = x[BatchKey.pv].shape[1]
        hist_pv = x[BatchKey.pv][:, : pv_t0_idx + 1]  # Shape: example, time, n_pv_systems

        # Reshape so each PV system is seen as a different example:
        hist_pv = einops.rearrange(
            hist_pv, "example time n_pv_systems -> (example n_pv_systems) time 1"
        )
        sat_trans_pv_attn_out = einops.rearrange(
            sat_trans_pv_attn_out,
            "example time n_pv_systems d_model -> (example n_pv_systems) time d_model",
            n_pv_systems=N_PV_SYSTEMS_PER_EXAMPLE,  # sanity check
            time=num_pv_timesteps,
        )
        hist_sat_trans_pv_attn_out = sat_trans_pv_attn_out[:, : pv_t0_idx + 1]

        pv_rnn_hist_enc_in = torch.concat((hist_pv, hist_sat_trans_pv_attn_out), dim=2)
        pv_rnn_hist_enc_in = pv_rnn_hist_enc_in.nan_to_num(0)
        pv_rnn_hist_enc_out, pv_rnn_hist_enc_hidden = self.pv_rnn_history_encoder(
            pv_rnn_hist_enc_in
        )

        # Now for the pv_rnn_future_decoder:
        future_sat_trans_pv_attn_out = sat_trans_pv_attn_out[:, pv_t0_idx + 1 :]
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
            d_model=self.hidden_size,
            time=num_pv_timesteps,
        )
        return pv_rnn_out


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class FullModel(pl.LightningModule):
    d_model: int = D_MODEL
    num_heads: int = N_HEADS
    dropout: float = 0.1
    share_weights_across_latent_transformer_layers: bool = False
    num_latent_transformer_encoders: int = 8
    cheat: bool = False  #: Use real satellite imagery of the future.
    #: Compute the loss on a central crop of the imagery.
    num_5_min_history_timesteps_during_training: Optional[int] = 4
    num_5_min_forecast_timesteps_during_training: Optional[int] = 6
    num_gaussians: int = 4
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
                    "/home/jack/dev/ocf/power_perceiver/power_perceiver/experiments/"
                    "power_perceiver/3qvkf1dy/checkpoints/"
                    "epoch=170-step=175104-just-satellite-predictor.state_dict.pth"
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

        self.nwp_processor = NWPProcessor(n_channels=len(NWP_CHANNELS))

        # Find temporal features, and help calibrate predictions using recent history:
        self.pv_rnn = PVRNN(
            hidden_size=self.d_model,
            num_layers=self.num_rnn_layers,
            dropout=self.dropout,
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

    def _predict_satellite(self, x: dict[BatchKey, torch.Tensor]) -> dict[BatchKey, torch.Tensor]:
        """Predict future satellite images. The SatellitePredictor always gets every timestep."""
        if self.cheat:
            predicted = x[BatchKey.hrvsatellite_actual][:, :NUM_HIST_SAT_IMAGES, 0]
        else:
            predicted = self.satellite_predictor(x=x)  # Shape: example, time, y, x

        assert predicted.isfinite().all()
        x[BatchKey.hrvsatellite_predicted] = predicted
        return x

    def _select_random_subset_of_timesteps(
        self, x: dict[BatchKey, torch.Tensor]
    ) -> dict[BatchKey, torch.Tensor]:
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
        x[BatchKey.hrvsatellite_actual] = x[BatchKey.hrvsatellite_actual][
            :, random_history_timestep_indexes
        ]
        x[BatchKey.hrvsatellite_predicted] = x[BatchKey.hrvsatellite_predicted][
            :, random_forecast_timestep_indexes - NUM_HIST_SAT_IMAGES
        ]
        for batch_key in (
            # If we go back to training the u-net, then we'll have to save `hrvsatellite`
            # before it is subsampled. See issue #156.
            # Don't include BatchKey.hrvsatellite_time_utc because all it's used for
            # is plotting satellite predictions, and we need all timesteps for that!
            # Subselect pv_time_utc here, so `plot_probabability_timeseries.plot_pv_power`
            # works correctly.
            # We *do* subset hrvsatellite_time_utc_fourier because it's used in the
            # satellite_transformer.
            BatchKey.hrvsatellite_time_utc_fourier,
            BatchKey.hrvsatellite_solar_azimuth,
            BatchKey.hrvsatellite_solar_elevation,
            BatchKey.pv,
            BatchKey.pv_time_utc,
            BatchKey.pv_time_utc_fourier,
            BatchKey.pv_solar_azimuth,
            BatchKey.pv_solar_elevation,
            BatchKey.gsp_5_min,
            BatchKey.gsp_5_min_time_utc_fourier,
            BatchKey.gsp_5_min_solar_azimuth,
            BatchKey.gsp_5_min_solar_elevation,
        ):
            x[batch_key] = x[batch_key][:, random_timestep_indexes]

        for batch_key in (
            BatchKey.hrvsatellite_t0_idx,
            BatchKey.pv_t0_idx,
            BatchKey.gsp_5_min_t0_idx,
        ):
            x[batch_key] = self.num_5_min_history_timesteps_during_training - 1

        return x

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Predict future satellite images. The SatellitePredictor always gets every timestep.
        x = self._predict_satellite(x)

        # Select a subset of 5-minute timesteps during training:
        if self.num_5_min_history_timesteps_during_training and self.training:
            x = self._select_random_subset_of_timesteps(x)
            num_5_min_timesteps = (
                self.num_5_min_history_timesteps_during_training
                + self.num_5_min_forecast_timesteps_during_training
            )
        else:
            num_5_min_timesteps = NUM_HIST_SAT_IMAGES + NUM_FUTURE_SAT_IMAGES

        x = _crop_satellite_and_spatial_coords(x)

        # Pass through the transformer!
        sat_trans_out = self.satellite_transformer(x=x)

        # Pass through PV RNN:
        sat_trans_pv_attn_out = sat_trans_out["pv_attn_out"]
        # Shape of `sat_trans_pv_attn_out`: (example time n_pv_systems d_model)
        pv_rnn_out = self.pv_rnn(x=x, sat_trans_pv_attn_out=sat_trans_pv_attn_out)
        del sat_trans_pv_attn_out
        n_pv_elements = pv_rnn_out.shape[1]

        # ----------------- Prepare inputs for the `time_transformer` -----------------------------

        # Get GSP attention outputs. Each GSP attention output timestep is seen as a separate
        # element into the `time transformer`. Remember, at this stage, `sat_trans_gsp_attn_out`
        # is 5-minutely. Shape (after squeeze): example time d_model.
        sat_trans_gsp_attn_out = sat_trans_out["gsp_attn_out"].squeeze()

        # Pad with zeros
        pv_rnn_out = maybe_pad_with_zeros(pv_rnn_out, requested_dim=self.d_model)
        sat_trans_gsp_attn_out = maybe_pad_with_zeros(
            sat_trans_gsp_attn_out, requested_dim=self.d_model
        )

        # Get GSP query for the time_transformer:
        # The query for the time_transformer is just for the half-hourly timesteps
        # (not the 5-minutely GSP queries used in the `SatelliteTransformer`.)
        gsp_query_generator: GSPQueryGenerator = self.satellite_transformer.gsp_query_generator
        gsp_query = gsp_query_generator.forward(
            x,
            include_history=True,
            base_batch_key=BatchKey.gsp,
            do_reshape_time_as_batch=False,
        )
        gsp_query = maybe_pad_with_zeros(gsp_query, requested_dim=self.d_model)

        # Prepare NWP inputs
        nwp_query = self.nwp_processor(x)
        nwp_query = maybe_pad_with_zeros(nwp_query, requested_dim=self.d_model)

        # Concatenate all the things we're going to feed into the "time transformer":
        # Shape: (example, elements, d_model)
        # `pv_rnn_out` must be the first set of elements.
        # `gsp_query` must be the last set of elements.
        # The shape is (example, query_elements, d_model).
        time_attn_in = torch.concat(
            (pv_rnn_out, sat_trans_gsp_attn_out, nwp_query, gsp_query), dim=1
        )

        # It's necessary to mask the time_transformer because some examples, where we're trying to
        # predict GSP PV power, might have NaN PV data (because the GSP is so far north it has no
        # PV systems!), and we don't want the attention mechanism to pay any attention to the
        # zero'd out (previously NaN) PV data when predicting GSP PV power.
        mask = time_attn_in.isnan().any(dim=2)
        time_attn_in = time_attn_in.nan_to_num(0)
        time_attn_out = self.time_transformer(time_attn_in, src_key_padding_mask=mask)

        # ---------------------------- MIXTURE DENSITY NETWORK -----------------------------------
        # The MDN doesn't like NaNs, and I think there will be NaNs when, for example,
        # the example is missing PV data:
        time_attn_out = time_attn_out.nan_to_num(0)
        predicted_pv_power = self.pv_mixture_density_net(time_attn_out[:, :n_pv_elements])

        # Reshape the PV power predictions
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

        return dict(
            predicted_pv_power=predicted_pv_power,  # Shape: (example time n_pv_sys mdn_features)
            predicted_gsp_power=predicted_gsp_power,  # Shape: (example time mdn_features)
            predicted_sat=x[BatchKey.hrvsatellite_predicted],  # Shape: example, time, y, x
            pv_time_utc=x[BatchKey.pv_time_utc],  # Give the subsampled times to plotting functions.
        )

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
        pv_t0_idx = batch[BatchKey.pv_t0_idx]
        pv_slice_from_t0 = slice(pv_t0_idx + 1, None)
        pv_mask_from_t0 = pv_mask[:, pv_slice_from_t0]

        gsp_mask = actual_gsp_power.isfinite()
        gsp_t0_idx = batch[BatchKey.gsp_t0_idx]
        gsp_slice_from_t0 = slice(gsp_t0_idx + 1, None)
        gsp_slice_from_t0_to_1h = slice(gsp_t0_idx + 1, gsp_t0_idx + 3)
        gsp_mask_from_t0 = gsp_mask[:, gsp_slice_from_t0]
        gsp_mask_from_t0_to_1h = gsp_mask[:, gsp_slice_from_t0_to_1h]

        predicted_pv_power_from_t0 = predicted_pv_power[:, pv_slice_from_t0][pv_mask_from_t0]
        actual_pv_power_from_t0 = actual_pv_power[:, pv_slice_from_t0][pv_mask_from_t0]

        predicted_gsp_power_from_t0 = predicted_gsp_power[:, gsp_slice_from_t0][gsp_mask_from_t0]
        actual_gsp_power_from_t0 = actual_gsp_power[:, gsp_slice_from_t0][gsp_mask_from_t0]

        predicted_gsp_power_from_t0_to_1h = predicted_gsp_power[:, gsp_slice_from_t0_to_1h][
            gsp_mask_from_t0_to_1h
        ]
        actual_gsp_power_from_t0_to_1h = actual_gsp_power[:, gsp_slice_from_t0_to_1h][
            gsp_mask_from_t0_to_1h
        ]

        # PV negative log prob loss:
        pv_distribution = get_distribution(predicted_pv_power_from_t0)
        pv_neg_log_prob_loss = -pv_distribution.log_prob(actual_pv_power_from_t0).mean()
        self.log(f"{self.tag}/pv_neg_log_prob", pv_neg_log_prob_loss)

        # PV power loss:
        pv_mse_loss = F.mse_loss(pv_distribution.mean, actual_pv_power_from_t0)
        self.log(f"{self.tag}/pv_mse", pv_mse_loss)

        pv_nmae_loss = F.l1_loss(pv_distribution.mean, actual_pv_power_from_t0)
        self.log(f"{self.tag}/pv_nmae", pv_nmae_loss)

        # GSP negative log prob loss from t0:
        gsp_distribution_from_t0 = get_distribution(predicted_gsp_power_from_t0)
        gsp_neg_log_prob_loss_from_t0 = -gsp_distribution_from_t0.log_prob(
            actual_gsp_power_from_t0
        ).mean()
        self.log(f"{self.tag}/gsp_neg_log_prob", gsp_neg_log_prob_loss_from_t0)

        # GSP negative log prob loss from t0 to 1h forecast:
        gsp_distribution_from_t0_to_1h = get_distribution(predicted_gsp_power_from_t0_to_1h)
        gsp_neg_log_prob_loss_from_t0_to_1h = -gsp_distribution_from_t0_to_1h.log_prob(
            actual_gsp_power_from_t0_to_1h
        ).mean()
        self.log(f"{self.tag}/gsp_neg_log_prob_from_t0_to_1h", gsp_neg_log_prob_loss_from_t0_to_1h)

        # GSP power loss:
        gsp_mse_loss = F.mse_loss(gsp_distribution_from_t0.mean, actual_gsp_power_from_t0)
        self.log(f"{self.tag}/gsp_mse", gsp_mse_loss)

        gsp_nmae_loss = F.l1_loss(gsp_distribution_from_t0.mean, actual_gsp_power_from_t0)
        self.log(f"{self.tag}/gsp_nmae", gsp_nmae_loss)

        # Total PV and GSP MSE loss:
        total_pv_and_gsp_mse_loss = gsp_mse_loss + pv_mse_loss
        self.log(f"{self.tag}/total_mse", total_pv_and_gsp_mse_loss)

        # Total NMAE loss:
        total_nmae_loss = pv_nmae_loss + gsp_nmae_loss
        self.log(f"{self.tag}/total_nmae", total_nmae_loss)

        # Total PV and GSP negative log probability loss:
        total_pv_and_gsp_neg_log_prob_loss = (
            gsp_neg_log_prob_loss_from_t0
            + (gsp_neg_log_prob_loss_from_t0_to_1h * 2)
            + pv_neg_log_prob_loss
        )
        self.log(
            f"{self.tag}/total_pv_and_gsp_neg_log_prob_loss", total_pv_and_gsp_neg_log_prob_loss
        )

        return {
            "loss": total_pv_and_gsp_neg_log_prob_loss,
            "predicted_gsp_power": predicted_gsp_power,
            "predicted_gsp_power_mean": get_distribution(predicted_gsp_power).mean,
            "actual_gsp_power": actual_gsp_power,
            "gsp_time_utc": batch[BatchKey.gsp_time_utc],
            "actual_pv_power": actual_pv_power,
            "predicted_pv_power": predicted_pv_power,
            "predicted_pv_power_mean": get_distribution(predicted_pv_power).mean,
            "pv_time_utc": network_out["pv_time_utc"],
            "predicted_sat": network_out["predicted_sat"],  # Shape: example, time, y, x
            "actual_sat": batch[BatchKey.hrvsatellite_actual][:, NUM_HIST_SAT_IMAGES:, 0],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

        def _lr_lambda(epoch):
            return 50 / (epoch + 50)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, verbose=True)
        return [optimizer], [scheduler]


# ---------------------------------- Training ----------------------------------

if __name__ == "__main__":
    # Important to seed the models when using DistributedDataProcessing, so the
    # models on different GPUs are initialised the same way. But we *do* want
    # to seed our data loader workers differently for each GPU, so we do that
    # in our own worker_init_fn.
    pl.seed_everything(42)

    if DEBUG:
        train_time_periods = pd.DataFrame({"start_dt": "2020-01-01", "end_dt": "2020-03-01"})
        val_time_periods = pd.DataFrame({"start_dt": "2021-01-01", "end_dt": "2021-03-01"})
    else:
        # Train on all of 2020. Plus every 5th week of 2021.
        every_week = pd.period_range("2020-01-01", "2021-12-31", freq="W")
        val_weeks = every_week[53::5]
        train_weeks = set(every_week) - set(val_weeks)
        train_weeks = pd.PeriodIndex(train_weeks).sort_values()

        # Now convert to our "time_periods" DataFrame:
        train_time_periods = pandas_periods_to_our_periods_dt(train_weeks)
        val_time_periods = pandas_periods_to_our_periods_dt(val_weeks)

    # Get data loaders
    train_dataloader = get_dataloader(
        time_periods=train_time_periods,
        num_workers=0 if DEBUG else 4,
        n_batches_per_epoch_per_worker=64 if DEBUG else 1024,
        load_subset_every_epoch=True,
        train=True,
    )

    N_GSPS_AFTER_FILTERING = 313
    val_dataloader = get_dataloader(
        time_periods=val_time_periods,
        # num_workers for NationalPVDataset MUST BE SAME 1!
        # OTHERWISE LogNationalPV BREAKS! See:
        # https://github.com/openclimatefix/power_perceiver/issues/130
        num_workers=0 if DEBUG else 1,
        n_batches_per_epoch_per_worker=64 if DEBUG else N_GSPS_AFTER_FILTERING,
        load_subset_every_epoch=False,
        train=False,
    )

    # Init model:
    model = FullModel()

    if ENABLE_WANDB:
        wandb_logger = WandbLogger(
            name="027.08: Train on more data. GCP-1",
            project="power_perceiver",
            entity="openclimatefix",
            log_model=True,
        )
        callbacks = [
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
            LogNationalPV(),
        ]
    else:
        wandb_logger = False
        callbacks = None

    # WARNING: Don't run multiple GPUs in ipython.
    trainer = pl.Trainer(
        gpus=GPUS,
        strategy="ddp" if len(GPUS) > 1 else None,
        max_epochs=200,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
