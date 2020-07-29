from typing import Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .locked_dropout import LockedDropout
from .torch_rnn_wrapper import TorchRNNWrapper


class WeightDrop(nn.Module):
    """
    A module that wraps another weight in which some weights will be replaced by 0 during training.
    Note that this wrapper is dedicated to the classes inheriting from torch.nn.RNNBase.
    """

    def __init__(self, module: nn.Module, weight_names: List[str], p: float = 0.):
        super().__init__()
        self.module = module
        self._drop_rate = p
        self._weight_names = weight_names
        for w_name in self._weight_names:
            # Makes a copy of the weights of the selected weights.
            w = getattr(self.module, w_name)
            self.register_parameter(f'{w_name}_raw', nn.Parameter(w.data))
            self.module._parameters[w_name] = F.dropout(w, p=self._drop_rate, training=False)

    def _set_weights(self):
        """Apply dropout to the raw weights."""
        for w_name in self._weight_names:
            raw_w = getattr(self, f'{w_name}_raw')
            self.module._parameters[w_name] = F.dropout(raw_w, p=self._drop_rate, training=self.training)

    def forward(self, *args):
        self._set_weights()
        return self.module.forward(*args)

    def reset(self):
        for weight in self._weight_names:
            raw_w = getattr(self, f'{weight}_raw')
            self.module._parameters[weight] = F.dropout(raw_w, p=self._drop_rate, training=False)
        if hasattr(self.module, 'reset'):
            self.module.reset()


class StackedRNN(nn.Module):
    """
    LSTM layers with residual connections.
    Dropout is only applied to non-recurrent connections.
    The interface is supposed to be the same as the original pytorch implementations or the RNN family.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool = True,
                 batch_first: bool = False,
                 inter_layer_dropout: float = 0.0,
                 weight_dropout: float = 0.0,
                 bidirectional: bool = False,
                 residual: bool = False,
                 rnn_type: str = 'LSTM'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = inter_layer_dropout
        self.bidirectional = bidirectional

        assert rnn_type in ['RNN', 'LSTM', 'GRU'], f"{rnn} is currently not supported."
        self.rnn_type = rnn_type
        self.num_layers = num_layers

        self.bidirectional = bidirectional
        num_directions = int(bidirectional) + 1

        self._residual = residual

        rnn_layers = []
        for i in range(num_layers):
            layer_in_size = input_size if i == 0 else hidden_size * num_directions

            rnn_layers.append(getattr(nn, rnn_type)(input_size=layer_in_size,
                                                    hidden_size=hidden_size,
                                                    num_layers=1,
                                                    bias=bias,
                                                    batch_first=batch_first,
                                                    dropout=0.,
                                                    bidirectional=bidirectional))

        if weight_dropout:
            # weight dropout is only applied to the hidden-to-hidden weights
            rnn_layers = [WeightDrop(rnn_layer, weight_names=['weight_hh_l0'], p=weight_dropout)
                          for rnn_layer in rnn_layers]

        self._rnn_layers = torch.nn.ModuleList(rnn_layers)

        self._inter_layer_dropout = LockedDropout(locked_dim=int(batch_first), p=inter_layer_dropout)

    def forward(self,
                input_tensors: Union[torch.Tensor, PackedSequence],
                hx: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None):

        """
        Parameters
        ----------
        input_tensors : shape `(seq_len, batch, input_size)`
            tensor containing the features of the input sequence.
            The input can also be a packed variable length sequence.

        hx : shape `(num_layers * num_directions, batch, hidden_size)`
            tensor containing the initial hidden state for each element in the batch.
            Defaults to zero if not provided. If the RNN is bidirectional,
            num_directions should be 2, else it should be 1.

        Returns
        -------
        output_tensors : shape `(seq_len, batch, num_directions * hidden_size)`:
            tensor containing the output features h_t from the last layer of the RNN, for each t.
            If a :class:`torch.nn.utils.rnn.PackedSequence` has been given as the input,
            the output will also be a packed sequence.
              For the unpacked case, the directions can be separated
              using ``output.view(seq_len, batch, num_directions, hidden_size)``,
              with forward and backward being direction `0` and `1` respectively.

        hiddens_and_maybe_cells : shape `(num_layers * num_directions, batch, hidden_size)`
            tensor containing the hidden state for `t = seq_len`.
            If the RNN is LSTM, this is a tuple of two tensors, hidden states and cell states.
        """

        # Split hidden tensors into per-layer ones
        if hx is None:
            hx_layers = [None for _ in range(self.num_layers)]
        elif self.rnn_type == 'LSTM':
            hiddens, cells = [h.unsqueeze(1) for h in hx]
            hx_layers = [(hiddens[i], cells[i]) for i in range(self.num_layers)]
        else:
            hx_layers = hx.unsqueeze(1)

        hiddens_and_maybe_cells = []
        input_is_packed_sequence = False
        for i, (rnn_layer, hx_layer) in enumerate(zip(self._rnn_layers, hx_layers)):

            output_tensors, hidden = rnn_layer(input_tensors, hx_layer)
            hiddens_and_maybe_cells.append(hidden)

            # To apply additive residual connection or dropout, unpack sequence.
            if isinstance(input_tensors, PackedSequence):
                input_is_packed_sequence = True
                output_tensors, seq_lens = pad_packed_sequence(output_tensors, batch_first=self.batch_first)
                input_tensors, seq_lens = pad_packed_sequence(input_tensors, batch_first=self.batch_first)

            output_tensors = self._inter_layer_dropout(output_tensors)

            if i != 0 and self._residual:
                output_tensors = output_tensors + input_tensors

            if input_is_packed_sequence:
                output_tensors = pack_padded_sequence(output_tensors, seq_lens, batch_first=self.batch_first)

            input_tensors = output_tensors

        if self.rnn_type == 'LSTM':
            hiddens, cells = zip(*hiddens_and_maybe_cells)
            hiddens_and_maybe_cells = (torch.cat(hiddens, dim=0), torch.cat(cells, dim=0))
        else:
            hiddens_and_maybe_cells = torch.cat(hiddens_and_maybe_cells, dim=0)

        return output_tensors, hiddens_and_maybe_cells


class StackedRNNWrapper(TorchRNNWrapper):
    """
    Always batch-first (but hiddens are not).
    You don't have to pack sequence, just feed embeddings with masks.
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 inter_layer_dropout: float = 0.0,
                 weight_dropout: float = 0.0,
                 residual: Union[bool, List[bool]] = False,
                 bidirectional: bool = False,
                 rnn_type: str = 'LSTM'):
        super().__init__(input_size, hidden_size)

        assert rnn_type in ['RNN', 'LSTM', 'GRU'], f"{rnn_type} is currently not supported."
        self.rnn = StackedRNN(input_size, hidden_size,
                              num_layers=num_layers, bias=bias,
                              batch_first=True, inter_layer_dropout=inter_layer_dropout, weight_dropout=weight_dropout,
                              bidirectional=bidirectional, residual=residual, rnn_type=rnn_type)

        self._init_attrs(rnn_type)
