from dataclasses import dataclass

import einops
import torch
from ocf_datapipes.utils.consts import BatchKey
from torch import nn

N_PV_SYSTEMS_PER_EXAMPLE = 8


@dataclass(eq=False)
class PVRNN(nn.Module):
    hidden_size: int
    num_layers: int
    dropout: float

    def __post_init__(self):
        super().__init__()

        rnn_kwargs = dict(
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.pv_rnn_history_encoder = nn.RNN(
            # Each timestep of the encoder RNN receives the output of the satellite_transformer
            # (which is of size d_model per PV system and per timestep), for a single PV system,
            # plus one timestep of that PV system's history.
            input_size=self.hidden_size + 1,
            **rnn_kwargs,
        )
        self.pv_rnn_future_decoder = nn.RNN(
            # Each timestep of the decoder RNN receives the output of the satellite_transformer
            # (which is of size d_model per PV system and per timestep) for a single PV system.
            input_size=self.hidden_size,
            **rnn_kwargs,
        )

    def forward(
        self, x: dict[BatchKey, torch.Tensor], sat_trans_pv_attn_out: torch.Tensor
    ) -> torch.Tensor:
        # Prepare inputs for the pv_rnn_history_encoder:
        # Each timestep of the encoder RNN receives the output of the satellite_transformer
        # (which is of size d_model per PV system and per timestep), for a single PV system,
        # plus one timestep of that PV system's history.
        # Some of `x[BatchKey.pv]` will be NaN. But that's OK. We mask missing examples.
        # And use nan_to_num later in this function.
        pv_t0_idx = x[BatchKey.pv_t0_idx]
        num_pv_timesteps = x[BatchKey.pv].shape[1]
        hist_pv = x[BatchKey.pv][:, : pv_t0_idx + 1]  # Shape: example, time, n_pv_systems

        # Reshape so each PV system is seen as a different example:
        hist_pv = einops.rearrange(
            hist_pv, "example time n_pv_systems -> (example n_pv_systems) time 1"
        )
        sat_trans_pv_attn_out = einops.rearrange(
            sat_trans_pv_attn_out,
            "example time n_pv_systems d_model -> (example n_pv_systems) time d_model",
            n_pv_systems=N_PV_SYSTEMS_PER_EXAMPLE,  # sanity check
            time=num_pv_timesteps,
        )
        hist_sat_trans_pv_attn_out = sat_trans_pv_attn_out[:, : pv_t0_idx + 1]

        pv_rnn_hist_enc_in = torch.concat((hist_pv, hist_sat_trans_pv_attn_out), dim=2)
        pv_rnn_hist_enc_in = pv_rnn_hist_enc_in.nan_to_num(0)
        pv_rnn_hist_enc_out, pv_rnn_hist_enc_hidden = self.pv_rnn_history_encoder(
            pv_rnn_hist_enc_in
        )

        # Now for the pv_rnn_future_decoder:
        future_sat_trans_pv_attn_out = sat_trans_pv_attn_out[:, pv_t0_idx + 1 :]
        future_sat_trans_pv_attn_out = future_sat_trans_pv_attn_out.nan_to_num(0)
        pv_rnn_fut_dec_out, _ = self.pv_rnn_future_decoder(
            future_sat_trans_pv_attn_out,
            pv_rnn_hist_enc_hidden,
        )

        # Concatenate the output from the encoder and decoder RNNs, and reshape
        # so each timestep and each PV system is a separate element into `time_transformer`.
        pv_rnn_out = torch.concat((pv_rnn_hist_enc_out, pv_rnn_fut_dec_out), dim=1)
        # rnn_out shape: (example n_pv_systems), time, d_model
        assert pv_rnn_out.isfinite().all()
        pv_rnn_out = einops.rearrange(
            pv_rnn_out,
            "(example n_pv_systems) time d_model -> example (time n_pv_systems) d_model",
            n_pv_systems=N_PV_SYSTEMS_PER_EXAMPLE,
            d_model=self.hidden_size,
            time=num_pv_timesteps,
        )
        return pv_rnn_out
