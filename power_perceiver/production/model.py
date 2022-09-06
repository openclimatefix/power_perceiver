"""
U-Net followed by the Power Perceiver.

See this issue for a diagram: https://github.com/openclimatefix/power_perceiver/issues/54
"""

# General imports
import logging
from dataclasses import dataclass
from typing import Optional

import einops

# ML imports
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

# power_perceiver imports
from ocf_datapipes.utils.consts import BatchKey
from torch import nn

from power_perceiver.hub import NowcastingModelHubMixin
from power_perceiver.production.pvrnn import N_PV_SYSTEMS_PER_EXAMPLE, PVRNN
from power_perceiver.production.satellite_transformer import (
    BOTTOM_IDX,
    D_MODEL,
    LEFT_IDX,
    N_HEADS,
    NUM_FUTURE_SAT_IMAGES,
    NUM_HIST_SAT_IMAGES,
    RIGHT_IDX,
    SATELLITE_TRANSFORMER_IMAGE_SIZE_PIXELS,
    TOP_IDX,
    SatellitePredictor,
    SatelliteTransformer,
    maybe_pad_with_zeros,
)
from power_perceiver.pytorch_modules.mixture_density_network import (
    MixtureDensityNetwork,
    get_distribution,
)
from power_perceiver.pytorch_modules.nwp_processor import NWPProcessor
from power_perceiver.pytorch_modules.query_generator import GSPQueryGenerator
from power_perceiver.pytorch_modules.self_attention import MultiLayerTransformerEncoder
from power_perceiver.utils import assert_num_dims
from power_perceiver.xr_batch_processor.reduce_num_timesteps import random_int_without_replacement

logging.basicConfig()
_log = logging.getLogger("power_perceiver")
_log.setLevel(logging.DEBUG)


# SatellitePredictor options

# PowerPerceiver options

NWP_CHANNELS = ("t", "dswrf", "prate", "r", "si10", "vis", "lcc", "mcc", "hcc")

DEBUG = False
ENABLE_WANDB = False


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6


# ---------------------------------- SatelliteTransformer ----------------------------------


# Indexes for cropping the centre of the satellite image:


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
class FullModel(pl.LightningModule, NowcastingModelHubMixin):
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
        # self.satellite_predictor.load_state_dict(
        #    torch.load(
        #        (
        #            "/home/jack/dev/ocf/power_perceiver/power_perceiver/experiments/power_perceiver/3qvkf1dy/checkpoints/epoch=170-step=175104-just-satellite-predictor.state_dict.pth"
        #        )
        #        if ON_DONATELLO
        #        else (
        #            "/home/jack/model_params/satellite_predictor/epoch=170-step=175104-just-satellite-predictor.state_dict.pth"
        #        )
        #    )
        # )

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
        time_attn_out = self.time_transformer(time_attn_in.float(), src_key_padding_mask=mask)

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
        predicted_gsp_power = self.pv_mixture_density_net(
            time_attn_out[:, -n_gsp_elements:].float()
        )

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

    def load_model(
        self,
        local_filename: Optional[str] = None,
        use_hf: bool = True,
    ):
        """
        Load model weights
        """

        if use_hf:
            _log.debug('Loading mode from Hugging Face "openclimatefix/power_perceiver" ')
            model = FullModel.from_pretrained("openclimatefix/power_perceiver")
            _log.debug("Loading mode from Hugging Face: done")
            return model
        else:
            _log.debug(f"Loading model weights from {local_filename}")
            model = self.load_from_checkpoint(checkpoint_path=local_filename)
            _log.debug("Loading model weights: done")
            return model
