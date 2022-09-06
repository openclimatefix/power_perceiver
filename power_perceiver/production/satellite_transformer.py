from dataclasses import dataclass

import einops
import pytorch_lightning as pl
import torch
from ocf_datapipes.utils.consts import BatchKey
from pytorch_msssim import ms_ssim
from torch import nn
from torch.nn import functional as F

from power_perceiver.consts import X_OSGB_MEAN, X_OSGB_STD, Y_OSGB_MEAN, Y_OSGB_STD
from power_perceiver.load_prepared_batches.data_sources.satellite import SAT_MEAN, SAT_STD
from power_perceiver.pytorch_modules.query_generator import GSPQueryGenerator, PVQueryGenerator
from power_perceiver.pytorch_modules.satellite_predictor import XResUNet
from power_perceiver.pytorch_modules.satellite_processor import HRVSatelliteProcessor
from power_perceiver.pytorch_modules.self_attention import MultiLayerTransformerEncoder

NUM_HIST_SAT_IMAGES = 12  # v15 pre-prepared batches use 7
NUM_FUTURE_SAT_IMAGES = 25  # v15 pre-prepared batches use 24
USE_TOPOGRAPHY = True
USE_SUN_POSITION = True
SATELLITE_PREDICTOR_IMAGE_WIDTH_PIXELS = 256
SATELLITE_PREDICTOR_IMAGE_HEIGHT_PIXELS = 128
SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS = 64
D_MODEL = 256  # Must be divisible by N_HEADS
N_HEADS = 32


def get_osgb_coords_for_coord_conv(batch: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
    """Returns tensor of shape (example, 2, y, x)."""
    y_osgb = batch[BatchKey.hrvsatellite_y_osgb]
    x_osgb = batch[BatchKey.hrvsatellite_x_osgb]

    # Normalise:
    y_osgb = (y_osgb - Y_OSGB_MEAN) / Y_OSGB_STD
    x_osgb = (x_osgb - X_OSGB_MEAN) / X_OSGB_STD

    # Concat:
    return torch.stack((y_osgb, x_osgb), dim=1)


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
        predicted_sat = self.satellite_predictor(data.float())
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


def maybe_pad_with_zeros(tensor: torch.Tensor, requested_dim: int) -> torch.Tensor:
    num_zeros_to_pad = requested_dim - tensor.shape[-1]
    assert num_zeros_to_pad >= 0, f"{requested_dim=}, {tensor.shape=}"
    if num_zeros_to_pad > 0:
        zero_padding_shape = tensor.shape[:2] + (num_zeros_to_pad,)
        zero_padding = torch.zeros(*zero_padding_shape, dtype=tensor.dtype, device=tensor.device)
        tensor = torch.concat((tensor, zero_padding), dim=2)
    return tensor


LEFT_IDX = (SATELLITE_PREDICTOR_IMAGE_WIDTH_PIXELS // 2) - (
    SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS // 2
)
RIGHT_IDX = LEFT_IDX + SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS
TOP_IDX = (SATELLITE_PREDICTOR_IMAGE_HEIGHT_PIXELS // 2) - (
    SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS // 2
)
BOTTOM_IDX = TOP_IDX + SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS


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
        attn_output = attn_input + self.transformer_encoder(
            attn_input.float(), src_key_padding_mask=mask
        )

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
