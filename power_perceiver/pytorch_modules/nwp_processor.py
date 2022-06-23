from dataclasses import dataclass

import einops
import torch
import torch.nn.functional as F
from torch import nn

from power_perceiver.consts import PV_SPACER_LEN, SATELLITE_SPACER_LEN, BatchKey
from power_perceiver.pytorch_modules.utils import get_spacer_tensor


@dataclass(eq=False)
class NWPProcessor(nn.Module):
    n_channels: int  # Number of NWP channels.
    max_steps: int  # Maximum number of NWP steps.
    channel_id_dim: int = 16

    def __post_init__(self):
        super().__init__()
        self.learnable_channel_ids = nn.Parameter(torch.randn(self.n_channels, self.channel_id_dim))

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        """Return a byte array ready for Perceiver.

        Args:
            x: A batch with at least these BatchKeys:
                nwp
                nwp_step
                nwp_target_time_utc_fourier

        Returns:
            tensor of shape (example, time * channel, feature).
        """

        # Patch all positions into a single element:
        nwp = x[BatchKey.nwp]  # Shape: (example, time, channel, y, x)
        nwp = einops.rearrange(nwp, "example time channel y x -> example time channel (y x)")
        assert nwp.shape[2] == self.n_channels

        # Repeat time fourier over all channels
        time_fourier = x[BatchKey.nwp_target_time_utc_fourier]  # (example, time, n_features)
        time_fourier = einops.repeat(
            time_fourier,
            "example time features -> example time channel features",
            channel=self.n_channels,
        )

        # Repeat learnt embedding ID for each channel:
        channel_ids = einops.repeat(
            self.learnable_channel_ids,
            "channel features -> example time channel features",
            example=nwp.shape[0],
            time=nwp.shape[1],
        )

        # One-hot encoding for NWP step, and then repeat across channels:
        nwp_step = F.one_hot(x[BatchKey.nwp_step], num_classes=self.max_steps)
        nwp_step = einops.repeat(
            nwp_step,
            "example time features -> example time channel features",
            channel=self.n_channels,
        )

        # Sun position:
        def _get_solar_position(batch_key: BatchKey) -> torch.Tensor:
            return einops.repeat(
                x[batch_key],
                "example time -> example time channel 1",
                channel=self.n_channels,
            )

        solar_azimuth = _get_solar_position(BatchKey.nwp_target_time_solar_azimuth)
        solar_elevation = _get_solar_position(BatchKey.nwp_target_time_solar_elevation)

        time_fourier_t0_dummy = y_fourier_dummy = x_fourier_dummy = torch.zeros_like(time_fourier)
        # length = 20 for satellite +
        satellite_and_pv_spacer = get_spacer_tensor(
            template=time_fourier,
            length=SATELLITE_SPACER_LEN + PV_SPACER_LEN,
        )

        # Concatenate on the feature dimension (the last dim):
        nwp_query = torch.concat(
            (
                time_fourier,
                time_fourier_t0_dummy,
                solar_azimuth,
                solar_elevation,
                y_fourier_dummy,
                x_fourier_dummy,
                satellite_and_pv_spacer,
                channel_ids,
                nwp_step,
                nwp,
            ),
            dim=-1,
        )

        return einops.rearrange(
            nwp_query, "example time channel features -> example (time channel) features"
        )
