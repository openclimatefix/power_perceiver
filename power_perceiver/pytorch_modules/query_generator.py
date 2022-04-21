from dataclasses import dataclass

import einops
import torch
from torch import nn

from power_perceiver.consts import BatchKey


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`
@dataclass(eq=False)
class PVQueryGenerator(nn.Module):
    """Create a query from the locations of the PV systems."""

    pv_system_id_embedding_dim: int
    num_pv_systems: int = 1400  # TODO: Set this to the correct number!

    def __post_init__(self):
        super().__init__()

        self.pv_system_id_embedding = nn.Embedding(
            num_embeddings=self.num_pv_systems,
            embedding_dim=self.pv_system_id_embedding_dim,
        )

    def forward(self, x: dict[BatchKey, torch.Tensor], start_idx_5_min: int = 0) -> torch.Tensor:
        """The returned tensor is of shape (example, n_pv_systems, query_dim)"""

        y_fourier = x[BatchKey.pv_y_osgb_fourier]  # (example, n_pv_systems, fourier_features)
        x_fourier = x[BatchKey.pv_x_osgb_fourier]

        pv_system_row_number = x[BatchKey.pv_system_row_number]  # (example, n_pv_systems)
        pv_system_embedding = self.pv_system_id_embedding(pv_system_row_number)
        n_pv_systems = x[BatchKey.pv_x_osgb].shape[1]

        time_idx_5_min = 6 + start_idx_5_min
        assert time_idx_5_min < x[BatchKey.pv].shape[1]

        # Select the timestep:
        time_fourier = x[BatchKey.pv_time_utc_fourier]  # (example, time, n_fourier_features)
        time_fourier = time_fourier[:, time_idx_5_min]
        # Repeat across all PV systems:
        time_fourier = einops.repeat(
            time_fourier,
            "example features -> example n_pv_systems features",
            n_pv_systems=n_pv_systems,
        )

        # Reshape solar features to: (example, n_pv_systems, 1)
        def _repeat_solar_feature_over_pv_systems(solar_feature: torch.Tensor) -> torch.Tensor:
            # Select the timestep:
            solar_feature = solar_feature[:, time_idx_5_min]
            return einops.repeat(
                solar_feature,
                "example -> example n_pv_systems 1",
                n_pv_systems=n_pv_systems,
            )

        solar_azimuth = _repeat_solar_feature_over_pv_systems(x[BatchKey.solar_azimuth])
        solar_elevation = _repeat_solar_feature_over_pv_systems(x[BatchKey.solar_elevation])

        pv_system_query = torch.concat(
            (
                y_fourier,
                x_fourier,
                time_fourier,
                solar_azimuth,
                solar_elevation,
                pv_system_embedding,
            ),
            dim=2,
        )

        # Missing PV systems are represented as NaN in the fourier features. Fill these with zeros.
        # (We do this because we can't mask the *query*. Instead, we'll ignore missing PV
        # systems in the objective function.)
        pv_system_query = torch.nan_to_num(pv_system_query, nan=0.0)

        return pv_system_query


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`
@dataclass(eq=False)
class GSPQueryGenerator(nn.Module):
    """Create a GSP query."""

    gsp_id_embedding_dim: int
    num_gsp_systems: int = 350

    def __post_init__(self):
        super().__init__()

        self.gsp_id_embedding = nn.Embedding(
            num_embeddings=self.num_gsp_systems,
            embedding_dim=self.gsp_id_embedding_dim,
        )

    def forward(self, x: dict[BatchKey, torch.Tensor], time_idx_30_min: int = 0) -> torch.Tensor:
        """The returned tensor is of shape (example, 1, query_dim)"""
        assert time_idx_30_min < x[BatchKey.gsp].shape[1]

        y_fourier = x[BatchKey.gsp_y_osgb_fourier]  # (example, fourier_features)
        x_fourier = x[BatchKey.gsp_x_osgb_fourier]

        gsp_id = x[BatchKey.gsp_id]  # Shape: (example,)
        gsp_id_embedding = self.gsp_id_embedding(gsp_id)

        # Select the timestep:
        time_fourier = x[BatchKey.gsp_time_utc_fourier]  # (example, time, n_fourier_features)
        time_fourier = time_fourier[:, time_idx_30_min]  # (example, n_fourier_features)

        gsp_query = torch.concat(
            (
                y_fourier,
                x_fourier,
                time_fourier,
                gsp_id_embedding,
            ),
            dim=1,  # All the inputs are of shape (example, features)
        )

        gsp_query = einops.rearrange(gsp_query, "example features -> example 1 features")
        assert not torch.isnan(gsp_query).any()
        return gsp_query
