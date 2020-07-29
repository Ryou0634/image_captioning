from typing import Union, List

import torch
import torch.nn as nn

from allennlp.modules.attention import Attention, DotProductAttention
from allennlp.nn import util

from .rnn_decoder import RNNDecoder
from .sequence_decoder_base import SequenceDecoder


@SequenceDecoder.register("luong2015_decoder")
class Luong2015Decoder(RNNDecoder):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 attention: Attention = DotProductAttention(),
                 input_feeding: bool = True,
                 residual: Union[bool, List[bool]] = False,
                 inter_layer_dropout: float = 0.1,
                 weight_dropout: float = 0.0,
                 rnn: str = 'LSTM'):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         residual=residual,
                         inter_layer_dropout=inter_layer_dropout,
                         weight_dropout=weight_dropout,
                         rnn=rnn)

        self.attention = attention
        self.input_feeding = input_feeding
        self.fuse_attention = nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, embedded_inputs, state=None):
        state = state or {}
        state = self._expand_hiddens_if_needed(state)
        # prepare rnn_inputs
        if self.input_feeding:
            if 'feed_vec' not in state:
                batch_size, seq_len, enc_out_dim = state['encoder_output'].size()
                state['feed_vec'] = state['encoder_output'].new_zeros((batch_size, enc_out_dim))

            embedded_inputs = torch.cat([embedded_inputs, state['feed_vec']], dim=1)

        decoder_output, state = super().forward(embedded_inputs, state)

        # compute scaled_dot_product_attention vector
        attention_weights = self.attention(decoder_output, state['encoder_output'], state['source_mask'])
        state = self._store_step_tensor_to_state(state, attention_weights, "attention_weights")
        attention_vector = util.weighted_sum(state['encoder_output'], attention_weights)

        decoder_output_with_attn = self.fuse_attention(torch.cat([decoder_output, attention_vector], dim=1))
        decoder_output_with_attn = torch.tanh(decoder_output_with_attn)

        if self.input_feeding:
            state['feed_vec'] = decoder_output_with_attn

        return decoder_output_with_attn, state
