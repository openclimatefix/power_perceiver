from dataclasses import dataclass

import einops
import torch
from torch import nn

from power_perceiver.consts import BatchKey
from power_perceiver.pytorch_modules.utils import repeat_over_time


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
                hrvsatellite
                hrvsatellite_y_osgb_fourier
                hrvsatellite_x_osgb_fourier
                hrvsatellite_surface_height
                solar_azimuth
                solar_elevation

        Returns:
            tensor of shape ((example * time), (y * x), feature).
        """
        # Ignore the "channels" dimension because HRV is just a single channel:
        hrvsatellite = x[BatchKey.hrvsatellite][:, :, 0]

        # Repeat the fourier features for each timestep of each example:
        n_timesteps = hrvsatellite.shape[1]
        y_fourier, x_fourier, surface_height = repeat_over_time(
            x=x,
            batch_keys=(
                BatchKey.hrvsatellite_y_osgb_fourier,
                BatchKey.hrvsatellite_x_osgb_fourier,
                BatchKey.hrvsatellite_surface_height,
            ),
            n_timesteps=n_timesteps,
        )
        # y_fourier and x_fourier are now of shape (example, time, y, x, n_fourier_features).
        # surface_height is now of shape (example, time, y, x).

        surface_height = surface_height.unsqueeze(-1)
        # Now surface_height is of shape (example, time, y, x, 1).

        # Reshape solar features to shape: (example, time, y, x, 1):
        def _repeat_solar_feature_over_x_and_y(solar_feature: torch.Tensor) -> torch.Tensor:
            return einops.repeat(
                solar_feature,
                "example time -> example time y x 1",
                y=hrvsatellite.shape[2],
                x=hrvsatellite.shape[3],
            )

        solar_azimuth = _repeat_solar_feature_over_x_and_y(x[BatchKey.solar_azimuth])
        solar_elevation = _repeat_solar_feature_over_x_and_y(x[BatchKey.solar_elevation])

        # Concatenate spatial features and solar features onto satellite imagery:
        byte_array = torch.concat(
            (hrvsatellite, y_fourier, x_fourier, solar_azimuth, solar_elevation, surface_height),
            dim=-1,
        )

        # Reshape so each timestep is seen as a separate example, and the 2D image
        # is flattened into a 1D array.
        byte_array = einops.rearrange(
            byte_array,
            "example time y x feature -> (example time) (y x) feature",
        )

        return byte_array


# TODO: Test. At an absolute minimum, do something like this:
# HRVSatelliteProcessor()(batch).shape
