from typing import Dict

import torch
import torch.nn as nn

from allennlp.nn.util import add_positional_features

from models.submodules.transformer_submodules import MultiHeadedAttention, PositionwiseFeedForward, SublayerConnection
from models.submodules.utils import clone_modules

from .rnn_decoder import SequenceDecoder

from .transformer_decoder import TransformerDecoder, make_subsequent_mask


class SelfAttentionDecoderLayer(nn.Module):
    """SelfAttentionDecoder is made of self-attn and feed forward (defined below)"""

    def __init__(self,
                 size: int,
                 self_attn: MultiHeadedAttention,
                 feed_forward: PositionwiseFeedForward,
                 dropout: float = 0.1,
                 pre_norm: bool = True
                 ):
        super(SelfAttentionDecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone_modules(SublayerConnection(size, dropout, pre_norm=pre_norm), 2)

    def forward(self,
                embedding_sequence: torch.Tensor,
                target_mask: torch.Tensor = None,
                apply_target_subsequent_mask: bool = True):
        if apply_target_subsequent_mask:
            # shape : (target_sequence_length, target_sequence_length)
            target_subsequent_mask = make_subsequent_mask(embedding_sequence)

        embedding_sequence = self.sublayer[0](embedding_sequence,
                                              lambda x: self.self_attn(x, x, x,
                                                                       key_value_mask=target_mask,
                                                                       attention_mask=target_subsequent_mask))

        return self.sublayer[1](embedding_sequence, self.feed_forward)


@SequenceDecoder.register('self_attention_decoder')
class SelfAttentionDecoder(TransformerDecoder):
    """Generic num_layers layer decoder with masking."""

    @staticmethod
    def _get_layer(size, num_attention_heads, dropout, feedforward_hidden_dim, activation, pre_norm):
        return SelfAttentionDecoderLayer(size=size,
                                         self_attn=MultiHeadedAttention(num_heads=num_attention_heads,
                                                                        embed_size=size,
                                                                        dropout=dropout),
                                         feed_forward=PositionwiseFeedForward(model_size=size,
                                                                              hidden_dim=feedforward_hidden_dim,
                                                                              activation=activation,
                                                                              dropout=dropout),
                                         dropout=dropout,
                                         pre_norm=pre_norm)

    def _forward(self,
                 embedding_sequence: torch.Tensor,
                 state: Dict[str, torch.Tensor],
                 apply_target_subsequent_mask: bool = True) -> torch.Tensor:

        if self.use_position_encoding:
            embedding_sequence = add_positional_features(embedding_sequence)

        output = embedding_sequence
        for layer in self.layers:
            output = layer(output,
                           state['target_mask'] if 'target_mask' in state else None,
                           apply_target_subsequent_mask=apply_target_subsequent_mask)

        if self.pre_norm:
            # We need to add an additional function of layer normalization to the top layer
            # to prevent the excessively increased value caused by the sum of unnormalized output
            # (https://arxiv.org/pdf/1906.01787.pdf)
            output = self.output_layer_norm(output)

        return output
