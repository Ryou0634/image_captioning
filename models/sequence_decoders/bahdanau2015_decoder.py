from typing import Union, List

import torch
import torch.nn as nn

from allennlp.modules.attention import Attention, DotProductAttention
from allennlp.nn import util

from .rnn_decoder import RNNDecoder
from .sequence_decoder_base import SequenceDecoder


@SequenceDecoder.register("bahdanau2015_decoder")
class Bahdanau2015Decoder(RNNDecoder):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 attention: Attention = DotProductAttention(),
                 residual: Union[bool, List[bool]] = False,
                 inter_layer_dropout: float = 0.1,
                 weight_dropout: float = 0.0,
                 output_size: int = None,
                 rnn: str = 'LSTM'):
        super().__init__(input_size=input_size,
                         hidden_size=hidden_size,
                         num_layers=num_layers,
                         residual=residual,
                         inter_layer_dropout=inter_layer_dropout,
                         weight_dropout=weight_dropout,
                         rnn=rnn)

        self.attention = attention
        self.output_size = output_size
        if self.output_size is not None:
            self.linear = nn.Linear(self._hidden_size + self.get_input_dim(), self.output_size)

    def get_output_dim(self):
        if self.output_size is not None:
            return self.output_size
        else:
            return self._hidden_size + self.get_input_dim()

    def forward(self, embedded_inputs, state=None):
        state = state or {}
        state = self._expand_hiddens_if_needed(state)
        if "source_mask" in state:
            source_mask = state['source_mask']
        else:
            source_mask = None
        # use the last layer of the hidden state of decoder as query to compute scaled_dot_product_attention vector
        attention_weights = self.attention(state['hiddens'][:, -1], state['encoder_output'], source_mask)
        state = self._store_step_tensor_to_state(state, attention_weights, "attention_weights")
        attention_vector = util.weighted_sum(state['encoder_output'], attention_weights)

        embedded_attn_input = torch.cat([embedded_inputs, attention_vector], dim=1)

        rnn_output, state = super().forward(embedded_attn_input, state)

        decoder_output = torch.cat([embedded_inputs, attention_vector, rnn_output], dim=1)
        if self.output_size:
            decoder_output = self.linear(decoder_output)

        return decoder_output, state
