from dataclasses import dataclass

import einops
import torch
from ocf_datapipes.utils.consts import BatchKey
from torch import nn

from power_perceiver.consts import PV_SPACER_LEN, SATELLITE_SPACER_LEN
from power_perceiver.pytorch_modules.utils import get_spacer_tensor


@dataclass(eq=False)
class NWPProcessor(nn.Module):
    n_channels: int  # Number of NWP channels.
    channel_id_dim: int = 16

    def __post_init__(self):
        super().__init__()
        self.learnable_channel_ids = nn.Parameter(torch.randn(self.n_channels, self.channel_id_dim))

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        """Return a byte array ready for Perceiver.

        Args:
            x: A batch with at least these BatchKeys:
                nwp
                nwp_target_time_utc_fourier
                nwp_init_time_utc_fourier

        Returns:
            tensor of shape (example, time * channel, feature).
        """

        # Patch all positions into a single element:
        nwp = x[BatchKey.nwp]  # Shape: (example, time, channel, y, x)
        nwp = einops.rearrange(nwp, "example time channel y x -> example time channel (y x)")
        assert nwp.shape[2] == self.n_channels

        # Repeat time fourier over all channels
        def _get_time(batch_key: BatchKey) -> torch.Tensor:
            return einops.repeat(
                x[batch_key],
                "example time features -> example time channel features",
                channel=self.n_channels,
            )

        time_fourier = _get_time(
            BatchKey.nwp_target_time_utc_fourier
        )  # (example, time, n_features)
        init_time_fourier = _get_time(BatchKey.nwp_init_time_utc_fourier)

        # Repeat learnt embedding ID for each channel:
        channel_ids = einops.repeat(
            self.learnable_channel_ids,
            "channel features -> example time channel features",
            example=nwp.shape[0],
            time=nwp.shape[1],
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

        y_fourier_dummy = x_fourier_dummy = torch.zeros_like(time_fourier)
        # length = 20 for satellite +
        satellite_and_pv_spacer = get_spacer_tensor(
            template=time_fourier,
            length=SATELLITE_SPACER_LEN + PV_SPACER_LEN,
        )

        # Concatenate on the feature dimension (the last dim):
        nwp_query = torch.concat(
            (
                time_fourier,
                init_time_fourier,
                solar_azimuth,
                solar_elevation,
                y_fourier_dummy,
                x_fourier_dummy,
                satellite_and_pv_spacer,
                channel_ids,
                nwp,
            ),
            dim=-1,
        )

        return einops.rearrange(
            nwp_query, "example time channel features -> example (time channel) features"
        )
