from dataclasses import dataclass

import einops
import torch
from ocf_datapipes.utils.consts import BatchKey
from torch import nn

from power_perceiver.pytorch_modules.query_generator import reshape_time_as_batch
from power_perceiver.utils import assert_num_dims


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`
@dataclass(eq=False)
class HRVSatelliteProcessor(nn.Module):
    def __post_init__(self):
        super().__init__()

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        """Returns a byte array ready for Perceiver.

        Args:
            x: A batch with at least these BatchKeys:
                hrvsatellite_actual
                hrvsatellite_predicted
                hrvsatellite_time_utc_fourier
                hrvsatellite_y_osgb_fourier
                hrvsatellite_x_osgb_fourier
                hrvsatellite_surface_height
                hrvsatellite_solar_azimuth
                hrvsatellite_solar_elevation
                hrvsatellite_t0_idx
            hrvsatellite: shape (batch_size, y, x)  (timesteps have been folded into batch_size)

        Returns:
            tensor of shape (example, (y * x), (time * feature)).
        """
        # The strategy is to first get all the tensors into shape (example * x, y, x, features)
        # and then, at the end of the function, flatten y and x, so each position is seen
        # as a new element.

        # Combine actual (history) and predicted satellite:
        t0_idx = x[BatchKey.hrvsatellite_t0_idx]
        hrvsatellite = torch.concat(
            (
                x[BatchKey.hrvsatellite_actual][:, : t0_idx + 1, 0],
                x[BatchKey.hrvsatellite_predicted][
                    :, :-1
                ].detach(),  # Next 2 hours but has 25 timesteps, not 24
            ),
            dim=1,  # Concat on the time dimension.
        )
        n_timesteps = hrvsatellite.shape[1]

        # Reshape so each timestep is seen as a separate example!
        hrvsatellite = einops.rearrange(hrvsatellite, "example time ... -> (example time) ...")
        timeless_x = reshape_time_as_batch(
            x=x,
            batch_keys=(
                BatchKey.hrvsatellite_time_utc_fourier,
                BatchKey.hrvsatellite_solar_azimuth,
                BatchKey.hrvsatellite_solar_elevation,
            ),
        )

        # Patch the hrvsatellite
        PATCH_SIZE = 4
        hrvsatellite = einops.rearrange(
            hrvsatellite,
            "example (y y_patch) (x x_patch) -> example y x (y_patch x_patch)",
            y_patch=PATCH_SIZE,
            x_patch=PATCH_SIZE,
        )

        # Get position encodings:
        y_fourier = x[BatchKey.hrvsatellite_y_osgb_fourier]
        x_fourier = x[BatchKey.hrvsatellite_x_osgb_fourier]
        # y_fourier and x_fourier are now of shape (example, y, x, n_fourier_features).

        # Patch the position encodings
        def _reduce(tensor):
            return einops.reduce(
                tensor,
                "example (y y_patch) (x x_patch) ... -> example y x ...",
                "mean",
                y_patch=PATCH_SIZE,
                x_patch=PATCH_SIZE,
            )

        y_fourier = _reduce(y_fourier)
        x_fourier = _reduce(x_fourier)

        time_fourier = timeless_x[BatchKey.hrvsatellite_time_utc_fourier]
        # `time_fourier` is now shape: (example * time, n_features)
        time_fourier = einops.repeat(
            time_fourier,
            "example features -> example y x features",
            y=hrvsatellite.shape[1],
            x=hrvsatellite.shape[2],
        )

        time_fourier_t0 = x[BatchKey.hrvsatellite_time_utc_fourier_t0]
        time_fourier_t0 = einops.repeat(
            time_fourier_t0,
            "example features -> (example time) y x features",
            time=n_timesteps,
            y=hrvsatellite.shape[1],
            x=hrvsatellite.shape[2],
        )

        surface_height = x[BatchKey.hrvsatellite_surface_height]  # (example, y, x)
        surface_height = _reduce(surface_height)
        surface_height = surface_height.unsqueeze(-1)  # (example, y, x, 1)

        y_fourier = torch.repeat_interleave(y_fourier, repeats=n_timesteps, dim=0)
        x_fourier = torch.repeat_interleave(x_fourier, repeats=n_timesteps, dim=0)
        surface_height = torch.repeat_interleave(surface_height, repeats=n_timesteps, dim=0)

        # Reshape solar features to shape: (example, y, x, 1):
        def _repeat_solar_feature_over_x_and_y(feature: torch.Tensor) -> torch.Tensor:
            # Select the last timestep:
            assert_num_dims(feature, 1)
            return einops.repeat(
                feature,
                "example -> example y x 1",
                y=hrvsatellite.shape[1],
                x=hrvsatellite.shape[2],
            )

        solar_azimuth = _repeat_solar_feature_over_x_and_y(
            timeless_x[BatchKey.hrvsatellite_solar_azimuth]
        )
        solar_elevation = _repeat_solar_feature_over_x_and_y(
            timeless_x[BatchKey.hrvsatellite_solar_elevation]
        )

        # Concatenate spatial features and solar features onto satellite imagery:
        # The shape of each tensor, and the concatenated `byte_array`, should be:
        # example * time, y, x, feature
        byte_array = torch.concat(
            (
                time_fourier,
                time_fourier_t0,
                solar_azimuth,
                solar_elevation,
                y_fourier,
                x_fourier,
                surface_height,
                hrvsatellite,
            ),
            dim=3,
        )

        # Reshape so each location is seen as a separate element.
        byte_array = einops.rearrange(
            byte_array,
            "example y x feature -> example (y x) feature",
        )

        return byte_array


# TODO: Test. At an absolute minimum, do something like this:
# HRVSatelliteProcessor()(batch).shape
