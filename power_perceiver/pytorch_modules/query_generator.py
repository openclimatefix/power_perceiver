from dataclasses import InitVar, dataclass
from typing import Optional, Sequence

import einops
import numpy as np
import torch
from torch import nn

from power_perceiver.consts import BatchKey
from power_perceiver.utils import assert_num_dims

# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`


@dataclass(eq=False)
class PVQueryGenerator(nn.Module):
    """Create a query from the locations of the PV systems."""

    # This must be an InitVar because PyTorch does not allow modules to be
    # assigned before super().__init__()
    pv_system_id_embedding: InitVar[nn.Embedding]
    num_gsps: int = 360  # Used to make sure PV IDs don't clash with GSP IDs!

    def __post_init__(self, pv_system_id_embedding):
        super().__init__()
        self.pv_system_id_embedding = pv_system_id_embedding

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        """The returned tensor is of shape (example * time, n_pv_systems, query_dim)."""
        timeless_x = reshape_time_as_batch(
            x,
            batch_keys=(
                BatchKey.pv,
                BatchKey.pv_solar_azimuth,
                BatchKey.pv_solar_elevation,
                BatchKey.pv_time_utc_fourier,
            ),
            set_to_nan_after_t0_idx=x[BatchKey.pv_t0_idx],
        )

        n_timesteps, n_pv_systems = x[BatchKey.pv].shape[1:]

        y_fourier = x[BatchKey.pv_y_osgb_fourier]  # (example, n_pv_systems, fourier_features)
        x_fourier = x[BatchKey.pv_x_osgb_fourier]

        pv_system_row_number = x[BatchKey.pv_system_row_number]  # (example, n_pv_systems)
        pv_system_row_number = pv_system_row_number + self.num_gsps
        pv_system_embedding = self.pv_system_id_embedding(pv_system_row_number.nan_to_num(0).int())

        time_fourier = timeless_x[
            BatchKey.pv_time_utc_fourier
        ]  # shape: ((example * time) features)
        assert_num_dims(time_fourier, 2)
        time_fourier_t0 = x[BatchKey.pv_time_utc_fourier_t0]  # shape: (example, features)

        # Repeat y_fourier, x_fourier, and pv_system_embedding across each timestep:
        y_fourier = y_fourier.repeat_interleave(repeats=n_timesteps, dim=0)
        x_fourier = x_fourier.repeat_interleave(repeats=n_timesteps, dim=0)
        pv_system_embedding = pv_system_embedding.repeat_interleave(repeats=n_timesteps, dim=0)
        time_fourier_t0 = time_fourier_t0.repeat_interleave(repeats=n_timesteps, dim=0)

        # Repeat across all PV systems:
        time_fourier = einops.repeat(
            time_fourier,
            "example features -> example n_pv_systems features",
            n_pv_systems=n_pv_systems,
        )
        time_fourier_t0 = einops.repeat(
            time_fourier_t0,
            "example features -> example n_pv_systems features",
            n_pv_systems=n_pv_systems,
        )

        # Reshape solar features to: (example, n_pv_systems, 1)
        def _repeat_solar_feature_over_pv_systems(solar_feature: torch.Tensor) -> torch.Tensor:
            # Check that the solar feature has been reshaped from `example time -> (example time)`.
            assert_num_dims(solar_feature, 1)
            return einops.repeat(
                solar_feature,
                "example -> example n_pv_systems 1",
                n_pv_systems=n_pv_systems,
            )

        solar_azimuth = _repeat_solar_feature_over_pv_systems(timeless_x[BatchKey.pv_solar_azimuth])
        solar_elevation = _repeat_solar_feature_over_pv_systems(
            timeless_x[BatchKey.pv_solar_elevation]
        )

        pv_power = timeless_x[BatchKey.pv].unsqueeze(-1)  # Shape: ((example * time) n_pv_systems 1)

        # The first element of dim 3 is zero for PV and one to mark that "this is GSP":
        pv_marker = torch.zeros_like(solar_azimuth)

        return torch.concat(
            (
                pv_marker,
                y_fourier,
                x_fourier,
                time_fourier,
                time_fourier_t0,  # So the model can tell which "step" this is.
                solar_azimuth,
                solar_elevation,
                pv_system_embedding,
                pv_power,
            ),
            dim=2,
        )


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`
@dataclass(eq=False)
class GSPQueryGenerator(nn.Module):
    """Create a GSP query."""

    # This must be an InitVar because PyTorch does not allow modules to be
    # assigned before super().__init__()
    gsp_id_embedding: InitVar[nn.Embedding]

    def __post_init__(self, gsp_id_embedding):
        super().__init__()
        self.gsp_id_embedding = gsp_id_embedding

    def forward(
        self,
        x: dict[BatchKey, torch.Tensor],
        include_history: bool,
        base_batch_key: BatchKey,
        do_reshape_time_as_batch: bool,
    ) -> torch.Tensor:
        """Create query for GSP PV forecasts.

        Args:
            x: The batch. Requires BatchKeys: gsp, gsp_y_osgb_fourier, gsp_x_osgb_fourier,
                gsp_id, gsp_time_utc_fourier, solar_azimuth, gsp_t0_idx
            base_batch_key: Either BatchKey.gsp or BatchKey.gsp_5_min.

        Returns tensor of shape (example * time, 1, query_dim) if do_reshape_time_as_batch,
        else (example, time, query_dim).
        """
        n_timesteps = x[base_batch_key].shape[1]

        def _get_batch_key(suffix: str) -> BatchKey:
            return BatchKey[base_batch_key.name + suffix]

        # Get tuple of BatchKeys to reshape each timestep as a new example:
        time_utc_fourier_batch_key = _get_batch_key("_time_utc_fourier")
        solar_az_batch_key = _get_batch_key("_solar_azimuth")
        solar_el_batch_key = _get_batch_key("_solar_elevation")

        if do_reshape_time_as_batch:
            t0_idx_batch_key = _get_batch_key("_t0_idx")
            t0_idx = x[t0_idx_batch_key]

            batch_keys = (
                base_batch_key,
                time_utc_fourier_batch_key,
                solar_az_batch_key,
                solar_el_batch_key,
            )

            timeless_x = reshape_time_as_batch(
                x,
                batch_keys=batch_keys,
                set_to_nan_after_t0_idx=t0_idx if include_history else None,
            )

        # gsp_{y,x}_osgb_fourier starts as shape (example, 1, fourier_features).
        y_fourier = x[BatchKey.gsp_y_osgb_fourier]
        x_fourier = x[BatchKey.gsp_x_osgb_fourier]

        gsp_id = x[BatchKey.gsp_id].squeeze()  # Shape: (example,)
        gsp_id_embedding = self.gsp_id_embedding(torch.nan_to_num(gsp_id, nan=0).int())

        if do_reshape_time_as_batch:
            time_fourier = timeless_x[time_utc_fourier_batch_key].unsqueeze(1)
            solar_azimuth = timeless_x[solar_az_batch_key].unsqueeze(-1).unsqueeze(-1)
            solar_elevation = timeless_x[solar_el_batch_key].unsqueeze(-1).unsqueeze(-1)
        else:
            time_fourier = x[time_utc_fourier_batch_key]
            solar_azimuth = x[solar_az_batch_key].unsqueeze(-1)
            solar_elevation = x[solar_el_batch_key].unsqueeze(-1)

        # shape of time_fourier is now: (example * time) 1 features
        assert_num_dims(time_fourier, 3)
        time_fourier_t0 = x[BatchKey.gsp_time_utc_fourier_t0]  # shape: (example, features)

        # Repeat y_fourier, x_fourier, and gsp_id_embedding across each timestep:
        y_fourier = y_fourier.repeat_interleave(repeats=n_timesteps, dim=0)
        x_fourier = x_fourier.repeat_interleave(repeats=n_timesteps, dim=0)
        gsp_id_embedding = gsp_id_embedding.repeat_interleave(repeats=n_timesteps, dim=0).unsqueeze(
            1
        )
        time_fourier_t0 = time_fourier_t0.repeat_interleave(repeats=n_timesteps, dim=0).unsqueeze(1)

        # The first element of dim 3 is zero for PV and one to mark that "this is GSP":
        gsp_marker = torch.ones_like(solar_azimuth)

        gsp_query_tuple = (
            gsp_marker,
            y_fourier,
            x_fourier,
            time_fourier,
            time_fourier_t0,
            solar_azimuth,
            solar_elevation,
            gsp_id_embedding,
        )
        if include_history:
            gsp_query_tuple += (timeless_x[base_batch_key].unsqueeze(-1),)

        gsp_query = torch.concat(gsp_query_tuple, dim=2)

        return gsp_query


def reshape_time_as_batch(
    x: dict[BatchKey, torch.Tensor],
    batch_keys: Sequence[BatchKey],
    set_to_nan_after_t0_idx: Optional[int] = None,
) -> dict[BatchKey, torch.Tensor]:
    new_batch: dict[BatchKey, torch.Tensor] = {}
    for batch_key in batch_keys:
        tensor = x[batch_key]
        if set_to_nan_after_t0_idx is not None:
            tensor = tensor.detach().clone()
            tensor[:, set_to_nan_after_t0_idx + 1 :] = np.NaN
        new_batch[batch_key] = einops.rearrange(tensor, "example time ... -> (example time) ...")
    return new_batch
