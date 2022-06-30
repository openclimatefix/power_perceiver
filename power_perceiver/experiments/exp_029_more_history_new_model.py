"""
U-Net followed by the Power Perceiver.

See this issue for a diagram: https://github.com/openclimatefix/power_perceiver/issues/54
"""

# General imports
import datetime
import logging
import socket
from dataclasses import dataclass

import einops

# ML imports
import matplotlib.pyplot as plt
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
from power_perceiver.consts import X_OSGB_MEAN, X_OSGB_STD, Y_OSGB_MEAN, Y_OSGB_STD, BatchKey
from power_perceiver.load_prepared_batches.data_sources.satellite import SAT_MEAN, SAT_STD
from power_perceiver.load_raw.data_sources.raw_gsp_data_source import RawGSPDataSource
from power_perceiver.load_raw.data_sources.raw_nwp_data_source import RawNWPDataSource
from power_perceiver.load_raw.data_sources.raw_pv_data_source import RawPVDataSource
from power_perceiver.load_raw.data_sources.raw_satellite_data_source import RawSatelliteDataSource
from power_perceiver.load_raw.national_pv_dataset import NationalPVDataset
from power_perceiver.load_raw.raw_dataset import RawDataset
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
from power_perceiver.pytorch_modules.query_generator import GSPQueryGenerator, PVQueryGenerator
from power_perceiver.pytorch_modules.satellite_predictor import XResUNet
from power_perceiver.pytorch_modules.satellite_processor import HRVSatelliteProcessor
from power_perceiver.pytorch_modules.self_attention import MultiLayerTransformerEncoder
from power_perceiver.transforms.nwp import NWPInterpolate
from power_perceiver.transforms.pv import PVDownsample

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
USE_SATELLITE = False
USE_NWP = False

SATELLITE_PREDICTOR_IMAGE_WIDTH_PIXELS = 256
SATELLITE_PREDICTOR_IMAGE_HEIGHT_PIXELS = 128

SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS = 64
N_PV_SYSTEMS_PER_EXAMPLE = 8

# PowerPerceiver options
D_MODEL = 128
N_HEADS = 16

ON_DONATELLO = socket.gethostname() == "donatello"

DEBUG = True
ENABLE_WANDB = False

if DEBUG:
    GPUS = [0]
elif ON_DONATELLO:
    GPUS = [0, 1, 2, 4]
else:  # On GCP
    GPUS = [0, 1]


# Important to seed the models when using DistributedDataProcessing, so the
# models on different GPUs are initialised the same way. But we *do* want
# to seed our data loader workers differently for each GPU, so we do that
# in our own worker_init_fn.
pl.seed_everything(42)


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
    )

    if USE_SATELLITE:
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
            history_duration=datetime.timedelta(hours=1),
            forecast_duration=datetime.timedelta(hours=2),
            **data_source_kwargs,
        )

    pv_data_source = RawPVDataSource(
        pv_power_filename="~/data/PV/Passiv/ocf_formatted/v0/passiv.netcdf",
        pv_metadata_filename="~/data/PV/Passiv/ocf_formatted/v0/system_metadata_OCF_ONLY.csv",
        roi_height_meters=96_000,
        roi_width_meters=96_000,
        n_pv_systems_per_example=N_PV_SYSTEMS_PER_EXAMPLE,
        transforms=[PVDownsample(freq="30T", expect_dataset=False)],
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=8),
        **data_source_kwargs,
    )

    gsp_data_source = RawGSPDataSource(
        gsp_pv_power_zarr_path="~/data/PV/GSP/v3/pv_gsp.zarr",
        gsp_id_to_region_id_filename="~/data/PV/GSP/eso_metadata.csv",
        sheffield_solar_region_path="~/data/PV/GSP/gsp_shape",
        history_duration=datetime.timedelta(hours=1),
        forecast_duration=datetime.timedelta(hours=8),
        **data_source_kwargs,
    )

    if USE_NWP:
        nwp_data_source = RawNWPDataSource(
            zarr_path=(
                (
                    "/mnt/storage_ssd_4tb/data/ocf/solar_pv_nowcasting/nowcasting_dataset_pipeline/"
                    "NWP/UK_Met_Office/UKV/zarr/UKV_intermediate_version_3.zarr"
                )
                if ON_DONATELLO
                else (
                    "gs://solar-pv-nowcasting-data/NWP/UK_Met_Office/"
                    "UKV_intermediate_version_3.zarr"
                )
            ),
            roi_height_pixels=4,
            roi_width_pixels=4,
            y_coarsen=16,
            x_coarsen=16,
            channels=["dswrf", "t", "si10", "prate"],
            history_duration=datetime.timedelta(hours=1),
            forecast_duration=datetime.timedelta(hours=8),
            transforms=[NWPInterpolate(freq="30T")],
            **data_source_kwargs,
        )

    np_batch_processors = [EncodeSpaceTime(), SaveT0Time()]
    if USE_SUN_POSITION:
        np_batch_processors.append(SunPosition(modality_name="gsp"))
        np_batch_processors.append(SunPosition(modality_name="pv"))
        if USE_SATELLITE:
            np_batch_processors.append(SunPosition(modality_name="hrvsatellite"))
    if USE_SATELLITE and USE_TOPOGRAPHY:
        np_batch_processors.append(Topography("/home/jack/europe_dem_2km_osgb.tif"))

    if USE_SATELLITE:
        # Delete imagery of the future, because we're not training the U-Net,
        # and we want to save GPU RAM.
        # But we do want hrvsatellite_time_utc to continue into the future by 2 hours because
        # downstream code relies on hrvsatellite_time_utc.
        # This must come last.
        np_batch_processors.append(
            DeleteForecastSatelliteImagery(num_hist_sat_images=NUM_HIST_SAT_IMAGES)
        )

    gsp_pv_maybe_nwp_maybe_sat = (gsp_data_source, pv_data_source)
    if USE_NWP:
        gsp_pv_maybe_nwp_maybe_sat += (nwp_data_source,)
    if USE_SATELLITE:
        gsp_pv_maybe_nwp_maybe_sat += (sat_data_source,)

    raw_dataset_kwargs = dict(
        n_examples_per_batch=32,
        n_batches_per_epoch=n_batches_per_epoch_per_worker,
        np_batch_processors=np_batch_processors,
        load_subset_every_epoch=load_subset_every_epoch,
        min_duration_to_load_per_epoch=datetime.timedelta(
            hours=48 if DEBUG else ((12 * 32) if ON_DONATELLO else (12 * 24))
        ),
        data_source_combos=dict(
            gsp_pv_maybe_nwp_maybe_sat=gsp_pv_maybe_nwp_maybe_sat,
        ),
        t0_freq="30T",
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


train_dataloader = get_dataloader(
    start_date="2020-01-01",
    end_date="2020-03-01" if DEBUG else "2020-12-31",
    num_workers=0 if DEBUG else 4,
    n_batches_per_epoch_per_worker=64 if DEBUG else 512,
    load_subset_every_epoch=True,
    train=True,
)

N_GSPS_AFTER_FILTERING = 313
val_dataloader = get_dataloader(
    start_date="2021-01-01",
    end_date="2021-03-01" if DEBUG else "2021-12-31",
    # num_workers for NationalPVDataset MUST BE SAME 1!
    # OTHERWISE LogNationalPV BREAKS! See:
    # https://github.com/openclimatefix/power_perceiver/issues/130
    num_workers=0 if DEBUG else 1,
    n_batches_per_epoch_per_worker=64 if DEBUG else N_GSPS_AFTER_FILTERING,
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


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class SingleTimestepTransformer(nn.Module):
    """Infers power for a single timestep, and a single target at a time."""

    d_model: int = D_MODEL  # `d_model` must be divisible by `num_heads`.
    pv_system_id_embedding_dim: int = 16
    num_heads: int = N_HEADS
    dropout: float = 0.1
    share_weights_across_latent_transformer_layers: bool = False
    num_latent_transformer_encoders: int = 16

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

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        # Reshape so each timestep is considered a different example:
        num_examples, num_timesteps = x[BatchKey.pv].shape[:2]

        # Process satellite data and queries:
        pv_context = self.pv_query_generator(x)
        gsp_context = self.gsp_query_generator(x, include_history=True)
        gsp_target = self.gsp_query_generator(x, include_history=False)

        # Pad with zeros if necessary to get up to self.d_model:
        pv_context = maybe_pad_with_zeros(pv_context, requested_dim=self.d_model)
        gsp_context = maybe_pad_with_zeros(gsp_context, requested_dim=self.d_model)
        gsp_target = maybe_pad_with_zeros(gsp_target, requested_dim=self.d_model)

        # Prepare inputs for the transformer_encoder:
        attn_input = (gsp_target, gsp_context, pv_context)

        if USE_SATELLITE:
            satellite_data = self.hrvsatellite_processor(x)
            satellite_data = maybe_pad_with_zeros(satellite_data, requested_dim=self.d_model)
            attn_input += (satellite_data,)

        attn_input = torch.concat(attn_input, dim=1)

        # Mask the NaN GSP and PV queries. True or non-zero value indicates value will be ignored.
        mask = attn_input.isnan().any(dim=2)
        attn_input = attn_input.nan_to_num(0)

        # Pass data into transformer_encoder. attn_output will be of shape:
        # (example * time, elements, d_model)
        attn_output = attn_input + self.transformer_encoder(attn_input, src_key_padding_mask=mask)

        # We only take the first element (which is the element for the target sequence)
        target_output = attn_output[:, 0]  # Shape (example * time, d_model)
        assert target_output.isfinite().all()

        # Reshape to (example time element d_model):
        target_output = einops.rearrange(
            target_output,
            "(example time) ... -> example time ...",
            example=num_examples,
            time=num_timesteps,
        )

        return target_output  # shape: example, time, d_model


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
@dataclass(eq=False)
class FullModel(pl.LightningModule):
    d_model: int = D_MODEL
    num_heads: int = N_HEADS
    dropout: float = 0.1
    share_weights_across_latent_transformer_layers: bool = False
    num_latent_transformer_encoders: int = 8
    cheat: bool = False  #: Use real satellite imagery of the future.
    num_gaussians: int = 2

    def __post_init__(self):
        super().__init__()

        if self.cheat:
            _log.warning("CHEATING MODE ENABLED! Using real satellite imagery of future!")

        # Predict future satellite images:
        if USE_SATELLITE:
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
        self.single_timestep_transformer = SingleTimestepTransformer()

        # TODO: RNN for target. In a separate nn.Module. See experiment 27.

        # Look across timesteps to produce the final output.
        # self.time_transformer = MultiLayerTransformerEncoder(
        #     d_model=self.d_model,
        #     num_heads=self.num_heads,
        #     dropout=self.dropout,
        #     share_weights_across_latent_transformer_layers=(
        #         self.share_weights_across_latent_transformer_layers
        #     ),
        #     num_latent_transformer_encoders=self.num_latent_transformer_encoders,
        # )

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

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        if USE_SATELLITE:
            x = self._predict_satellite(x)

        single_step_trans_out = self.single_timestep_transformer(x=x)
        # Shape: (example time d_model)

        # The MDN doesn't like NaNs:
        single_step_trans_out = single_step_trans_out.nan_to_num(0)
        return self.pv_mixture_density_net(single_step_trans_out)

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

        mdn_predicted_power = self(batch)

        # GSP LOSS ######################
        actual_gsp_power = batch[BatchKey.gsp].squeeze()

        # Mask predicted and actual PV and GSP (some examples don't have PV and/or GSP)
        # For more discussion of how to mask losses in pytorch, see:
        # https://discuss.pytorch.org/t/masking-input-to-loss-function/121830/3
        gsp_t0_idx = batch[BatchKey.gsp_t0_idx]
        gsp_mask = actual_gsp_power.isfinite()
        gsp_mask_from_t0 = gsp_mask[:, gsp_t0_idx + 1 :]

        predicted_gsp_power_from_t0 = mdn_predicted_power[:, gsp_t0_idx + 1 :][gsp_mask_from_t0]
        actual_gsp_power_from_t0 = actual_gsp_power[:, gsp_t0_idx + 1 :][gsp_mask_from_t0]

        # GSP negative log prob loss:
        gsp_distribution_masked = get_distribution(predicted_gsp_power_from_t0)
        gsp_neg_log_prob_loss = -gsp_distribution_masked.log_prob(actual_gsp_power_from_t0).mean()
        self.log(f"{self.tag}/gsp_neg_log_prob", gsp_neg_log_prob_loss)

        # GSP power loss:
        gsp_mse_loss = F.mse_loss(gsp_distribution_masked.mean, actual_gsp_power_from_t0)
        self.log(f"{self.tag}/gsp_mse", gsp_mse_loss)

        gsp_nmae_loss = F.l1_loss(gsp_distribution_masked.mean, actual_gsp_power_from_t0)
        self.log(f"{self.tag}/gsp_nmae", gsp_nmae_loss)

        return {
            "loss": gsp_neg_log_prob_loss,
            "predicted_gsp_power": mdn_predicted_power,
            "predicted_gsp_power_mean": get_distribution(mdn_predicted_power).mean,
            "actual_gsp_power": actual_gsp_power,
            "gsp_time_utc": batch[BatchKey.gsp_time_utc],
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)

        def _lr_lambda(epoch):
            return 50 / (epoch + 50)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda, verbose=True)
        return [optimizer], [scheduler]


# ---------------------------------- Training ----------------------------------

model = FullModel()

if ENABLE_WANDB:
    wandb_logger = WandbLogger(
        name=(
            "029.00: New model. No RNN yet. 8 hr GSP fcst."
            " num_latent_transformer_encoders=8. GCP-1."
        ),
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
        LogSatellitePlots(),
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
