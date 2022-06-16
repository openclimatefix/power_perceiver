from dataclasses import InitVar, dataclass

import einops
import torch
from torch import nn

from power_perceiver.consts import T0_IDX_30_MIN, BatchKey
from power_perceiver.utils import assert_num_dims

# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`


@dataclass(eq=False)
class PVQueryGenerator(nn.Module):
    """Create a query from the locations of the PV systems.

    This returns n_pv_systems x n_timestep queries per example. In other words, each query
    is about a single PV system at a single timestep.
    """

    # This must be an InitVar because PyTorch does not allow modules to be
    # assigned before super().__init__()
    pv_system_id_embedding: InitVar[nn.Embedding]
    num_gsps: int = 360  # Used to make sure PV IDs don't clash with GSP IDs!

    def __post_init__(self, pv_system_id_embedding):
        super().__init__()
        self.pv_system_id_embedding = pv_system_id_embedding

    def forward(self, x: dict[BatchKey, torch.Tensor]) -> torch.Tensor:
        """The returned tensor is of shape (example, n_pv_systems, query_dim).

        We assume the PVQueryGenerator is *only* used in SatelliteTransformer,
        and `pv_time_utc_fourier`, `solar_azimuth`, and `solar_elevation` will have already been
        reshaped to `example time -> (example * time)`.
        """
        y_fourier = x[BatchKey.pv_y_osgb_fourier]  # (example, n_pv_systems, fourier_features)
        x_fourier = x[BatchKey.pv_x_osgb_fourier]

        pv_system_row_number = x[BatchKey.pv_system_row_number]  # (example, n_pv_systems)
        pv_system_row_number = pv_system_row_number + self.num_gsps
        pv_system_embedding = self.pv_system_id_embedding(pv_system_row_number.nan_to_num(0).int())
        n_pv_systems = x[BatchKey.pv_x_osgb].shape[1]

        # (example features) if for_satellite_transformer else (example, time, n_fourier_features)
        time_fourier = x[BatchKey.pv_time_utc_fourier]  # shape: ((example * time) features)
        assert_num_dims(time_fourier, 2)
        time_fourier_t0 = x[BatchKey.pv_time_utc_fourier_t0]  # shape: (orig_examples, features)

        # Repeat y_fourier, x_fourier, and pv_system_embedding across each timestep:
        n_repeats = int(time_fourier.shape[0] / y_fourier.shape[0])
        y_fourier = y_fourier.repeat_interleave(repeats=n_repeats, dim=0)
        x_fourier = x_fourier.repeat_interleave(repeats=n_repeats, dim=0)
        pv_system_embedding = pv_system_embedding.repeat_interleave(repeats=n_repeats, dim=0)
        time_fourier_t0 = time_fourier_t0.repeat_interleave(repeats=n_repeats, dim=0)

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

        solar_azimuth = _repeat_solar_feature_over_pv_systems(
            x[BatchKey.hrvsatellite_solar_azimuth]
        )
        solar_elevation = _repeat_solar_feature_over_pv_systems(
            x[BatchKey.hrvsatellite_solar_elevation]
        )

        # The first element of dim 3 is zero for PV and one to mark that "this is GSP":
        pv_marker = torch.zeros_like(y_fourier)

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
        self, x: dict[BatchKey, torch.Tensor], for_satellite_transformer: bool = True
    ) -> torch.Tensor:
        """Create query for GSP PV forecasts.

        Args:
            x: The batch. Requires BatchKeys: gsp, gsp_y_osgb_fourier, gsp_x_osgb_fourier,
                gsp_id, gsp_time_utc_fourier, solar_azimuth
            for_satellite_transformer: The query for the SatelliteTransformer uses 5-minutely
                data. The query for the time_transformer uses 30-minutely data.

        Returns tensor of shape (example, time, query_dim).
        """
        # gsp_{y,x}_osgb_fourier starts as shape (example, 1, fourier_features).
        y_fourier = x[BatchKey.gsp_y_osgb_fourier][:, 0]  # (example, fourier_features)
        x_fourier = x[BatchKey.gsp_x_osgb_fourier][:, 0]
        n_original_examples = y_fourier.shape[0]

        gsp_id = x[BatchKey.gsp_id].squeeze()  # Shape: (example,)
        gsp_id_embedding = self.gsp_id_embedding(torch.nan_to_num(gsp_id, nan=0).int())

        # The first element of dim 3 is zero for PV and one to mark that "this is GSP":
        gsp_marker = torch.ones_like(y_fourier)

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

        if for_satellite_transformer:
            # There might be NaNs in time_fourier.
            # NaNs will be masked in `SatelliteTransformer.forward`.
            # solar_azimuth is reshaped upstream.
            # TODO: Include "5-minutely" GSP time in the Satellite Transformer query.
            n_new_examples = x[BatchKey.hrvsatellite_solar_azimuth].shape[0]
            n_repeats = int(n_new_examples / n_original_examples)
            gsp_query = gsp_query.repeat_interleave(repeats=n_repeats, dim=0)
        else:
            time_fourier = x[BatchKey.gsp_time_utc_fourier]  # (example, time, n_fourier_features)
            assert_num_dims(time_fourier, 3)
            n_timesteps = time_fourier.shape[1]
            time_fourier_t0 = x[BatchKey.gsp_time_utc_fourier_t0]  # (example, n_fourier_features)
            # Repeat the existing query over every timestep
            time_fourier_t0 = einops.repeat(
                time_fourier_t0,
                "example features -> example time features",
                time=n_timesteps,
            )
            gsp_query = einops.repeat(
                gsp_query,
                "example 1 features -> example time features",
                time=n_timesteps,
            )
            # Get the recent history of GSP power: Take a copy because we modify the tensor:
            # See this discussion for why we use `tensor.detach().clone()` to copy the tensor:
            # https://stackoverflow.com/questions/55266154/pytorch-preferred-way-to-copy-a-tensor
            gsp_power = x[BatchKey.gsp].detach().clone()  # shape: batch, time, 1
            gsp_power[:, T0_IDX_30_MIN + 1 :] = 0
            gsp_history_mask = torch.ones_like(gsp_power)
            gsp_history_mask[:, T0_IDX_30_MIN + 1 :] = 0

            gsp_solar_elevation = x[BatchKey.gsp_solar_elevation]
            gsp_solar_azimuth = x[BatchKey.gsp_solar_azimuth]

            gsp_query = torch.concat(
                (
                    gsp_query,
                    time_fourier,
                    time_fourier_t0,
                    gsp_power,
                    gsp_history_mask,
                    gsp_solar_azimuth,
                    gsp_solar_elevation,
                ),
                dim=2,
            )

        return gsp_query
