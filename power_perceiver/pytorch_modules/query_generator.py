from dataclasses import dataclass

import einops
import torch
from torch import nn

from power_perceiver.consts import BatchKey


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`
@dataclass(eq=False)
class QueryGenerator(nn.Module):
    """Create a query using a learnt array and the locations of the PV systems."""

    num_fourier_features: int  # TOTAL (for y, x, and time_utc)
    pv_system_id_embedding_dim: int
    num_pv_systems: int = 1400  # TODO: Set this to the correct number!

    def __post_init__(self):
        super().__init__()
        # Plus two for solar azimuth and elevation
        self.query_dim = self.num_fourier_features + self.pv_system_id_embedding_dim + 2

        self.pv_system_id_embedding = nn.Embedding(
            num_embeddings=self.num_pv_systems,
            embedding_dim=self.pv_system_id_embedding_dim,
        )

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        y_fourier = x[BatchKey.pv_y_osgb_fourier]  # (example, n_pv_systems, fourier_features)
        x_fourier = x[BatchKey.pv_x_osgb_fourier]

        pv_system_row_number = x[BatchKey.pv_system_row_number]  # (example, n_pv_systems)
        pv_system_embedding = self.pv_system_id_embedding(pv_system_row_number)

        time_fourier = x[BatchKey.pv_time_utc_fourier]  # (example, time, n_fourier_features)
        # Select the last timestep:
        time_fourier = time_fourier[:, 12]
        # Repeat across all PV systems:
        n_pv_systems = x[BatchKey.pv_x_osgb].shape[1]
        time_fourier = einops.repeat(
            time_fourier,
            "example features -> example n_pv_systems features",
            n_pv_systems=n_pv_systems,
        )

        # Reshape solar features to: (example, n_pv_systems, 1)
        def _repeat_solar_feature_over_x_and_y(solar_feature: torch.Tensor) -> torch.Tensor:
            # Select the last timestep:
            solar_feature = solar_feature[:, 12]
            return einops.repeat(
                solar_feature,
                "example -> example n_pv_systems 1",
                n_pv_systems=n_pv_systems,
            )

        solar_azimuth = _repeat_solar_feature_over_x_and_y(x[BatchKey.solar_azimuth])
        solar_elevation = _repeat_solar_feature_over_x_and_y(x[BatchKey.solar_elevation])

        pv_system_query = torch.concat(
            (
                y_fourier,
                x_fourier,
                time_fourier,
                pv_system_embedding,
                solar_azimuth,
                solar_elevation,
            ),
            dim=-1,
        )

        # Missing PV systems are represented as NaN in the fourier features. Fill these with zeros.
        # (We do this because we can't mask the *query*. Instead, we'll ignore missing PV
        # systems in the objective function.)
        pv_system_query = torch.nan_to_num(pv_system_query, nan=0.0)

        return pv_system_query
