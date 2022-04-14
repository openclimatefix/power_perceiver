from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class TransformerEncoderLayerForVisualisingWeights(nn.TransformerEncoderLayer):
    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Self-attention (sa) block."""
        # This version of the `_sa_block` method is functionally identical to the
        # one in nn.TransformerEncoderLayer.
        # The only change we make is to pass in all parameters into `self.self_attn` as
        # *positional* parameters (not keyword) so the `forward_pre_hook`
        # can access and modify them :). Because the `forward_pre_hook`
        # can only access positional params. It can't access keyword params.
        # Then we can use the `forward_pre_hook` to change `need_weights` to True.
        need_weights = False
        x = self.self_attn(x, x, x, key_padding_mask, need_weights, attn_mask)[0]
        return self.dropout1(x)


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`
@dataclass(eq=False)
class MultiLayerTransformerEncoder(nn.Module):
    d_model: int
    num_heads: int = 8
    dropout: float = 0.0
    share_weights_across_latent_transformer_layers: bool = True
    num_latent_transformer_encoders: int = 4

    def __post_init__(self):
        super().__init__()

        # TransformerEncoderLayer is made up of self-attn and feedforward network.
        # This standard encoder layer is based on the paper “Attention Is All You Need”
        transformer_encoder_layer = TransformerEncoderLayerForVisualisingWeights(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
            activation="gelu",
        )

        if self.share_weights_across_latent_transformer_layers:
            transformer_encoder_layers = [
                transformer_encoder_layer
            ] * self.num_latent_transformer_encoders
            self.transformer_encoder = nn.Sequential(*transformer_encoder_layers)
        else:
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer=transformer_encoder_layer,
                num_layers=self.num_latent_transformer_encoders,
            )

    def forward(self, *args, **kwargs):
        return self.transformer_encoder(*args, **kwargs)


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`
@dataclass(eq=False)
class Perceiver(nn.Module):
    """
    Input and output tensors are provided as shape (batch, seq, feature).

    Init args:
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
    """

    query_dim: int
    byte_array_dim: int
    num_heads: int = 8
    dropout: float = 0.0
    share_weights_across_latent_transformer_layers: bool = True
    num_latent_transformer_encoders: int = 4
    need_weights: bool = False

    def __post_init__(self):
        super().__init__()
        assert self.num_latent_transformer_encoders >= 1

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.query_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True,
            kdim=self.byte_array_dim,
            vdim=self.byte_array_dim,
        )

        self.latent_transformer = MultiLayerTransformerEncoder(
            d_model=self.query_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            share_weights_across_latent_transformer_layers=(
                self.share_weights_across_latent_transformer_layers
            ),
            num_latent_transformer_encoders=self.num_latent_transformer_encoders,
        )

    def forward(
        self,
        query: torch.Tensor,
        byte_array: torch.Tensor,
        byte_array_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Query embeddings of shape `(N, L, query_dim)`, where `N` is the batch size,
                and `L` is the number of query elements (the target sequence length).
                Queries are compared against key-value pairs to produce the output.
            byte_array: Key embeddings of shape `(N, S, byte_array_dim)`, where `N` is the
                batch size, and `S` is the number of byte_array elements
                (the source sequence length).
            byte_array_padding_mask: If specified, a mask of shape `(N, S)` indicating which
                elements within ``byte_array`` to ignore for the purpose of attention
                (i.e. treat as "padding"). Shape should be `(N, S)`. Binary and byte masks
                are supported. For a binary mask, a ``True`` value indicates that the corresponding
                ``byte_array`` value will be ignored for the purpose of attention. For a byte mask,
                a non-zero value indicates that the corresponding ``byte_array`` value
                will be ignored.

        Returns:
            **attn_output** - Attention outputs of shape `(N, L, query_dim)`, where `N` is the
                batch size, and `L` is the number of query elements (the target sequence length).
        """
        attn_output, attn_weights = self.cross_attention(
            query=query,
            key=byte_array,
            value=byte_array,
            need_weights=self.need_weights,
            key_padding_mask=byte_array_padding_mask,
        )
        return self.latent_transformer(attn_output)


# See https://discuss.pytorch.org/t/typeerror-unhashable-type-for-my-torch-nn-module/109424/6
# for why we set `eq=False`
@dataclass(eq=False)
class PerceiverIO(nn.Module):
    encoder_query_dim: int
    decoder_query_dim: int
    byte_array_dim: int
    num_encoder_heads: int
    num_decoder_heads: int
    dropout: float = 0.0
    share_weights_across_latent_transformer_layers: bool = True
    num_latent_transformer_encoders: int = 4
    num_cross_attends: int = 1

    def __post_init__(self):
        super().__init__()
        assert self.num_cross_attends >= 1

        self.perceiver_encoders = nn.ModuleList()
        for _ in range(self.num_cross_attends):
            self.perceiver_encoders.append(
                Perceiver(
                    query_dim=self.encoder_query_dim,
                    byte_array_dim=self.byte_array_dim,
                    num_heads=self.num_encoder_heads,
                    dropout=self.dropout,
                    share_weights_across_latent_transformer_layers=(
                        self.share_weights_across_latent_transformer_layers
                    ),
                    num_latent_transformer_encoders=self.num_latent_transformer_encoders,
                )
            )

        self.decoder = nn.MultiheadAttention(
            embed_dim=self.decoder_query_dim,
            num_heads=self.num_decoder_heads,
            dropout=self.dropout,
            batch_first=True,
            kdim=self.encoder_query_dim,
            vdim=self.encoder_query_dim,
        )

    def forward(
        self,
        encoder_query: torch.Tensor,
        byte_array: torch.Tensor,
        decoder_query: torch.Tensor,
        byte_array_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Encoder
        encoder_output = encoder_query
        for perceiver_encoder in self.perceiver_encoders:
            encoder_output = perceiver_encoder(
                query=encoder_output,
                byte_array=byte_array,
                byte_array_padding_mask=byte_array_padding_mask,
            )

        # Decoder
        decoder_output, decoder_weights = self.decoder(
            query=decoder_query,
            key=encoder_output,
            value=encoder_output,
            need_weights=False,
        )
        return decoder_output
