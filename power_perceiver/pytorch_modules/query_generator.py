from dataclasses import InitVar, dataclass

import einops
import torch
from torch import nn

from power_perceiver.consts import BatchKey
from power_perceiver.utils import assert_num_dims

# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`


@dataclass(eq=False)
class PVQueryGenerator(nn.Module):
    """Create a query from the locations of the PV systems."""

    t0_idx_5_min: int

    # This must be an InitVar because PyTorch does not allow modules to be
    # assigned before super().__init__()
    pv_system_id_embedding: InitVar[nn.Embedding]
    num_gsps: int = 360  # Used to make sure PV IDs don't clash with GSP IDs!

    def __post_init__(self, pv_system_id_embedding):
        super().__init__()
        self.pv_system_id_embedding = pv_system_id_embedding

    def forward(
        self,
        x: dict[BatchKey, torch.Tensor],
        for_satellite_transformer: bool = True,
    ) -> torch.Tensor:
        """The returned tensor is of shape (example, n_pv_systems, query_dim)

        If `for_satellite_transformer` is True then any Tensor which originally had a
        time dimension will be assumed to have already been reshape from
        `example time -> (example * time)`.

        If `for_satellite_transformer` is False then the history of PV data will be included.
        """

        y_fourier = x[BatchKey.pv_y_osgb_fourier]  # (example, n_pv_systems, fourier_features)
        x_fourier = x[BatchKey.pv_x_osgb_fourier]

        pv_system_row_number = x[BatchKey.pv_system_row_number]  # (example, n_pv_systems)
        pv_system_row_number = pv_system_row_number + self.num_gsps
        pv_system_embedding = self.pv_system_id_embedding(
            torch.nan_to_num(pv_system_row_number, nan=0).int()
        )
        n_pv_systems = x[BatchKey.pv_x_osgb].shape[1]

        # (example features) if for_satellite_transformer else (example, time, n_fourier_features)
        time_fourier = x[BatchKey.pv_time_utc_fourier]
        if for_satellite_transformer:
            # time_fourier shape: (example features)
            assert_num_dims(time_fourier, 2)
            n_repeats = int(time_fourier.shape[0] / y_fourier.shape[0])
            y_fourier = torch.repeat_interleave(y_fourier, repeats=n_repeats, dim=0)
            x_fourier = torch.repeat_interleave(x_fourier, repeats=n_repeats, dim=0)
            pv_system_embedding = torch.repeat_interleave(
                pv_system_embedding, repeats=n_repeats, dim=0
            )
        else:
            # time_fourier shape: (example, time, n_fourier_features)
            assert_num_dims(time_fourier, 3)
            time_fourier = time_fourier[:, self.t0_idx_5_min]
        # Repeat across all PV systems:
        time_fourier = einops.repeat(
            time_fourier,
            "example features -> example n_pv_systems features",
            n_pv_systems=n_pv_systems,
        )

        # Reshape solar features to: (example, n_pv_systems, 1)
        def _repeat_solar_feature_over_pv_systems(solar_feature: torch.Tensor) -> torch.Tensor:
            if for_satellite_transformer:
                assert_num_dims(solar_feature, 1)
            else:
                assert_num_dims(solar_feature, 2)
                # Select the timestep. The original shape is (example, time).
                solar_feature = solar_feature[:, self.t0_idx_5_min]
            return einops.repeat(
                solar_feature,
                "example -> example n_pv_systems 1",
                n_pv_systems=n_pv_systems,
            )

        solar_azimuth = _repeat_solar_feature_over_pv_systems(x[BatchKey.solar_azimuth])
        solar_elevation = _repeat_solar_feature_over_pv_systems(x[BatchKey.solar_elevation])

        # The first element of dim 3 is zero for PV and one to mark that "this is GSP":
        pv_marker = torch.zeros_like(y_fourier)

        pv_system_query = torch.concat(
            (
                pv_marker,
                y_fourier,
                x_fourier,
                time_fourier,
                solar_azimuth,
                solar_elevation,
                pv_system_embedding,
            ),
            dim=2,
        )

        if not for_satellite_transformer:
            pv_power = x[BatchKey.pv]  # (batch_size, time, n_pv_systems)
            assert_num_dims(pv_power, 3)
            pv_power = pv_power[:, : self.t0_idx_5_min + 1]
            pv_power = einops.rearrange(
                pv_power, "example time n_pv_systems -> example n_pv_systems time"
            )
            pv_system_query = torch.concat((pv_system_query, pv_power), dim=2)

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

    # This must be an InitVar because PyTorch does not allow modules to be
    # assigned before super().__init__()
    gsp_id_embedding: InitVar[nn.Embedding]

    def __post_init__(self, gsp_id_embedding):
        super().__init__()
        self.gsp_id_embedding = gsp_id_embedding

    def forward(
        self, x: dict[BatchKey, torch.Tensor], for_satellite_transformer: bool = True
    ) -> torch.Tensor:
        """Create query for GSP PV forecasts.

        Args:
            x: The batch. Requires BatchKeys: gsp_y_osgb_fourier, gsp_x_osgb_fourier,
                gsp_id, gsp_time_utc_fourier, solar_azimuth
            for_satellite_transformer: The query for the SatelliteTransformer uses 5-minutely
                data. The query for the time_transformer uses 30-minutely data.

        Returns tensor of shape (example, time, query_dim).
        """
        # gsp_{y,x}_osgb_fourier starts as shape (example, 1, fourier_features).
        y_fourier = x[BatchKey.gsp_y_osgb_fourier][:, 0]  # (example, fourier_features)
        x_fourier = x[BatchKey.gsp_x_osgb_fourier][:, 0]
        n_original_examples = y_fourier.shape[0]

        gsp_id = x[BatchKey.gsp_id]  # Shape: (example,)
        gsp_id_embedding = self.gsp_id_embedding(torch.nan_to_num(gsp_id, nan=0).int())

        # The first element of dim 3 is zero for PV and one to mark that "this is GSP":
        gsp_marker = torch.ones_like(y_fourier)

        import ipdb; ipdb.set_trace()  # TODO: Remove!
        gsp_query = torch.concat(
            (
                gsp_marker,
                y_fourier,
                x_fourier,
                gsp_id_embedding,
            ),
            dim=1,  # All the inputs are of shape (example, features)
        )
        gsp_query = einops.rearrange(gsp_query, "example features -> example 1 features")
        assert not torch.isnan(gsp_query).any()

        if for_satellite_transformer:
            # There might be NaNs in time_fourier.
            # NaNs will be masked in `SatelliteTransformer.forward`.
            # solar_azimuth is reshaped upstream.
            n_new_examples = x[BatchKey.solar_azimuth].shape[0]
            n_repeats = int(n_new_examples / n_original_examples)
            gsp_query = torch.repeat_interleave(gsp_query, repeats=n_repeats, dim=0)
        else:
            time_fourier = x[BatchKey.gsp_time_utc_fourier]  # (example, time, n_fourier_features)
            assert_num_dims(time_fourier, 3)
            assert not torch.isnan(time_fourier).any()
            n_timesteps = time_fourier.shape[1]
            # Repeat the existing query over every timestep
            gsp_query = einops.repeat(
                gsp_query,
                "example 1 features -> example time features",
                time=n_timesteps,
            )
        gsp_query = torch.concat((gsp_query, time_fourier), dim=2)

        return gsp_query
