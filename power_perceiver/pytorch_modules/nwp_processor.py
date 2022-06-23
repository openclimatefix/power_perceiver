from dataclasses import dataclass

import einops
import torch
from torch import nn

from power_perceiver.consts import BatchKey


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

        Returns:
            tensor of shape (example, time, feature).
        """

        # Patch all positions into a single element:
        # TODO: Use proper encodings for each NWP channel?
        nwp = x[BatchKey.nwp]  # (example, time, channel, y, x)
        nwp = einops.rearrange(nwp, "example time channel y x -> example time channel (y x)")

        # Repeat time fourier over all channels
        assert nwp.shape[2] == self.n_channels
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

        # Concatenate time fourier on the final dimension:
        nwp_query = torch.concat((time_fourier, channel_ids, nwp), dim=-1).float()
        # Shape: example time channel features

        import ipdb

        ipdb.set_trace()

        return einops.rearrange(
            nwp_query, "example time channel features -> example (time channel) features"
        )
