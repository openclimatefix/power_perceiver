from dataclasses import dataclass

import einops
import torch
from torch import nn

from power_perceiver.consts import BatchKey
from power_perceiver.pytorch_modules.utils import repeat_over_time


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`
@dataclass(eq=False)
class QueryGenerator(nn.Module):
    """Create a query using a learnt array and the locations of the PV systems."""

    num_fourier_features: int  # TOTAL (for both x and y)
    num_elements_query_padding: int
    pv_system_id_embedding_dim: int
    num_pv_systems: int = 1400  # TODO: Set this to the correct number!

    def __post_init__(self):
        super().__init__()
        # Plus two for solar azimuth and elevation
        self.query_dim = self.num_fourier_features + self.pv_system_id_embedding_dim + 2

        self.query_padding = nn.Parameter(
            torch.randn(self.num_elements_query_padding, self.query_dim) / 5
        )
        self.pv_system_id_embedding = nn.Embedding(
            num_embeddings=self.num_pv_systems,
            embedding_dim=self.pv_system_id_embedding_dim,
        )

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        # Repeat the fourier features for each timestep of each example:
        n_timesteps = x[BatchKey.pv].shape[1]
        y_fourier, x_fourier, pv_system_row_number = repeat_over_time(
            x=x,
            batch_keys=(
                BatchKey.pv_y_osgb_fourier,
                BatchKey.pv_x_osgb_fourier,
                BatchKey.pv_system_row_number,
            ),
            n_timesteps=n_timesteps,
        )
        # y_fourier and x_fourier are now of shape (example, time, n_pv_systems, n_fourier_features)

        # Reshape solar features to: (example, time, n_pv_systems, 1)
        def _repeat_solar_feature_over_x_and_y(solar_feature: torch.Tensor) -> torch.Tensor:
            return einops.repeat(
                solar_feature,
                "example time -> example time n_pv_systems 1",
                n_pv_systems=x[BatchKey.pv_x_osgb].shape[1],
            )

        solar_azimuth = _repeat_solar_feature_over_x_and_y(x[BatchKey.solar_azimuth])
        solar_elevation = _repeat_solar_feature_over_x_and_y(x[BatchKey.solar_elevation])

        pv_system_embedding = self.pv_system_id_embedding(pv_system_row_number)

        pv_system_query = torch.concat(
            (y_fourier, x_fourier, pv_system_embedding, solar_azimuth, solar_elevation), dim=-1
        )
        del y_fourier, x_fourier, pv_system_embedding

        # Missing PV systems are represented as NaN in the fourier features. Fill these with zeros.
        # (We do this because we can't mask the *query*. Instead, we'll ignore missing PV
        # systems in the objective function.)
        pv_system_query = torch.nan_to_num(pv_system_query, nan=0.0)

        # Reshape so every timestep is a different example.
        pv_system_query = einops.rearrange(
            pv_system_query,
            "example time n_pv_systems query_dim -> (example time) n_pv_systems query_dim",
            query_dim=self.query_dim,
        )

        # Repeat the learnt query padding for every example in the batch:
        batch_size = pv_system_query.shape[0]
        batched_query_padding = einops.repeat(
            self.query_padding,
            "num_query_elements query_dim -> batch_size num_query_elements query_dim",
            batch_size=batch_size,
            query_dim=self.query_dim,
        )
        query = torch.concat((batched_query_padding, pv_system_query), dim=1)
        return query
