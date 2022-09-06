from typing import Iterable

import einops
import torch
from ocf_datapipes.utils.consts import BatchKey


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Adapted from https://discuss.pytorch.org/t/how-to-write-a-loss-function-with-mask/53461/4
    zeros = torch.tensor(0, dtype=tensor.dtype, device=tensor.device)
    masked_tensor = torch.where(mask, tensor, zeros)
    total = masked_tensor.sum()
    num_selected_elements = mask.sum()
    return total / num_selected_elements


def repeat_over_time(
    x: dict[BatchKey, torch.Tensor], batch_keys: Iterable[BatchKey], n_timesteps: int
) -> list[torch.Tensor]:
    repeated_tensors = []
    for batch_key in batch_keys:
        tensor = x[batch_key]
        tensor = einops.repeat(tensor, "batch_size ... -> batch_size time ...", time=n_timesteps)
        repeated_tensors.append(tensor)
    return repeated_tensors


def get_spacer_tensor(template: torch.Tensor, length: int) -> torch.Tensor:
    """Get a tensor of zeros."""
    shape = template.shape[:-1] + (length,)
    return torch.zeros(
        shape,
        dtype=torch.float32,
        device=template.device,
    )
