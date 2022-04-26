from dataclasses import dataclass

import einops
import torch
from torch import nn

from power_perceiver.consts import BatchKey
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
                hrvsatellite
                hrvsatellite_y_osgb_fourier
                hrvsatellite_x_osgb_fourier
                hrvsatellite_surface_height
                solar_azimuth
                solar_elevation

        Returns:
            tensor of shape (example, (y * x), (time * feature)).
        """
        # Ignore the "channels" dimension because HRV is just a single channel:
        hrvsatellite = x[BatchKey.hrvsatellite][:, :, 0]

        # Get position encodings:
        y_fourier = x[BatchKey.hrvsatellite_y_osgb_fourier]
        x_fourier = x[BatchKey.hrvsatellite_x_osgb_fourier]
        # y_fourier and x_fourier are now of shape (example, y, x, n_fourier_features).

        time_fourier = x[BatchKey.hrvsatellite_time_utc_fourier]  # (example, n_features)
        time_fourier = einops.repeat(
            time_fourier,
            "example features -> example y x features",
            y=hrvsatellite.shape[1],
            x=hrvsatellite.shape[2],
        )

        surface_height = x[BatchKey.hrvsatellite_surface_height]  # (example, y, x)
        surface_height = surface_height.unsqueeze(-1)  # (example, y, x, 1)

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

        solar_azimuth = _repeat_solar_feature_over_x_and_y(x[BatchKey.solar_azimuth])
        solar_elevation = _repeat_solar_feature_over_x_and_y(x[BatchKey.solar_elevation])

        # Concatenate spatial features and solar features onto satellite imagery:
        byte_array = torch.concat(
            (
                y_fourier,
                x_fourier,
                time_fourier,
                solar_azimuth,
                solar_elevation,
                surface_height,
                hrvsatellite,
            ),
            dim=-1,
        )

        # Reshape so each location is seen as a separate element.
        byte_array = einops.rearrange(
            byte_array,
            "example y x feature -> example (y x) feature",
        )

        return byte_array


# TODO: Test. At an absolute minimum, do something like this:
# HRVSatelliteProcessor()(batch).shape
