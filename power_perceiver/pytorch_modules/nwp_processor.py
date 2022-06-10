import einops
import numpy as np
import torch
from torch import nn

from power_perceiver.consts import BatchKey


class NWPProcessor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        """Return a byte array ready for Perceiver.

        Args:
            x: A batch with at least these BatchKeys:
                nwp
                nwp_target_time_utc_fourier

        Returns:
            tensor of shape (example, time, feature).
        """

        # Patch all positions and all channels into a single element:
        # TODO: Use proper encodings for each NWP channel?
        nwp = x[BatchKey.nwp]  # (example, time, channel, y, x)
        nwp = einops.rearrange(nwp, "example time channel y x -> example time (channel y x)")

        # Concatenate time fourier on the final dimension:
        time_fourier = x[BatchKey.nwp_target_time_utc_fourier]  # (example, time, n_features)
        time_fourier = time_fourier.astype(np.float32)
        return torch.concat((nwp, time_fourier), dim=-1)
