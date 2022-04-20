from dataclasses import dataclass

import einops
import torch
from torch import nn

from power_perceiver.consts import BatchKey


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`
@dataclass(eq=False)
class HRVSatelliteProcessor(nn.Module):
    def __post_init__(self):
        super().__init__()

    def forward(
        self,
        x: dict[BatchKey, torch.Tensor],
        start_idx: int = 0,
        start_idx_offset: int = 0,
        num_timesteps: int = 4,
        interval: int = 3,
        satellite_only: bool = False,
    ) -> torch.Tensor:
        """Returns a byte array ready for Perceiver.

        Args:
            x: A batch with at least these BatchKeys:
                hrvsatellite
                hrvsatellite_y_osgb_fourier
                hrvsatellite_x_osgb_fourier
                hrvsatellite_surface_height
                solar_azimuth
                solar_elevation
            start_idx: The index of the `t0` timestep.
            num_timesteps: The number of timesteps to include.
            interval: The interval (in number of timesteps) between each timestep.
                For example, an interval of 3 would be 15 minutes.
            satellite_only: If True, don't bother computing fourier features etc.

        Returns:
            tensor of shape (example, (y * x), (time * feature)).
        """
        # Ignore the "channels" dimension because HRV is just a single channel:
        hrvsatellite = x[BatchKey.hrvsatellite][:, :, 0]

        # Select four timesteps at 15-minute intervals, starting at start_idx.
        sat_start_idx = start_idx + start_idx_offset
        # + 1 because we want to *include* the last timestep:
        sat_end_idx = sat_start_idx + (num_timesteps * interval) + 1
        hrvsatellite = hrvsatellite[:, sat_start_idx:sat_end_idx:interval]
        assert (
            hrvsatellite.shape[1] == num_timesteps
        ), f"{hrvsatellite.shape[1]=} != {num_timesteps=}"

        # Reshape so each timestep is concatenated into the `patch` dimension:
        hrvsatellite = einops.rearrange(
            hrvsatellite,
            "example time y x feature -> example y x (time feature)",
            time=num_timesteps,
        )

        if satellite_only:
            # Reshape so each location is seen as a separate element.
            return einops.rearrange(
                hrvsatellite,
                "example y x feature -> example (y x) feature",
            )

        # Get position encodings:
        y_fourier = x[BatchKey.hrvsatellite_y_osgb_fourier]
        x_fourier = x[BatchKey.hrvsatellite_x_osgb_fourier]
        # y_fourier and x_fourier are now of shape (example, y, x, n_fourier_features).

        time_fourier = x[BatchKey.hrvsatellite_time_utc_fourier]  # (example, time, n_features)
        # Select the time encoding of the last timestep:
        time_fourier = time_fourier[:, sat_end_idx]  # (example, n_fourier_features)
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
            feature = feature[:, sat_end_idx]
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
