from typing import Union, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from allennlp.common import Registrable


class TorchRNNWrapper(nn.Module, Registrable):
    """
    Wrapper for torch RNN family classes.
    In the original pytorch implementation of RNNs, you have to sort, pack, and other stuffs with batched inputs,
    so this class will take the job for you and make life easier.
    Just feed batched sequences (usually padded) of embeddings with masks, without sorting!
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 bias: bool = True,
                 dropout: float = 0.0,
                 bidirectional: bool = False,
                 rnn_type: str = 'LSTM'):

        super().__init__()

        assert rnn_type in ['RNN', 'LSTM', 'GRU'], f"{rnn_type} is currently not supported."
        self.rnn = getattr(nn, rnn_type)(input_size, hidden_size,
                                         num_layers=num_layers, bias=bias,
                                         batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self._init_attrs(rnn_type)

    def _init_attrs(self, rnn_type: str):
        self.rnn_type = rnn_type
        self.input_size = self.rnn.input_size
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers
        self.bidirectional = self.rnn.bidirectional

    def get_output_dim(self):
        return self.rnn.hidden_size * (1 + int(self.rnn.bidirectional))

    @staticmethod
    def _get_permutation_indices(mask: torch.LongTensor):
        """
        Get the index for sorting with the length of the sequences.
        Empty sequences will be removed, but later restored with `restoration_idx`.
        """

        seq_lens = mask.long().sum(-1)

        sorted_seq_lens, perm_idx = seq_lens.sort(descending=True)

        # remove empty sequences
        num_non_zero_seqs = len(seq_lens.nonzero())
        truncated_sorted_seq_lens = sorted_seq_lens[:num_non_zero_seqs]
        truncated_perm_idx = perm_idx[:num_non_zero_seqs]

        # compute restoration index to sort tensors into the original order later.
        _, restoration_idx = perm_idx.sort()

        return truncated_perm_idx, truncated_sorted_seq_lens, restoration_idx

    def _perm_tensor(self,
                     tensors: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                     perm_index: torch.Tensor,
                     batch_dim: int = 0):
        """
        Permute `tensors` with `perm_index` along the `batch_dim` dimension.
        Note this function can also handle a tuple of tensors, such as (hiddnes, cells).
        """

        if isinstance(tensors, torch.Tensor):
            return tensors.index_select(dim=batch_dim, index=perm_index)

        else:
            return tuple(self._perm_tensor(t, perm_index, batch_dim) for t in tensors)

    def _restore_outputs(self,
                         tensors: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
                         restoration_idx: torch.LongTensor,
                         batch_dim: int = 0):
        """
        Restore tensors sorted along `batch_dim` dimension.
        If the original tensor contained empty sequences, which has been removed before computation,
        they will also be restored by `insert_zeros()`.
        """

        if isinstance(tensors, torch.Tensor):

            num_truncated = restoration_idx.size(0) - tensors.size(batch_dim)

            outputs = insert_zeros(tensors, target_dim=batch_dim, num_zeros=num_truncated)

            return outputs.index_select(dim=batch_dim, index=restoration_idx)  # (batch, seq_len, output_size)

        else:
            return tuple(self._restore_outputs(t, restoration_idx, batch_dim) for t in tensors)

    def forward(self,
                inputs: torch.Tensor,
                hiddens: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]] = None,
                mask: torch.LongTensor = None):
        """
        Compute batched outputs through all time steps.
        Inputs will be padded, and so is the outputs. You can un-pad the outputs with ``seq_lens``.

        Parameters :
        inputs : shape `(batch_size, max_seq_len, dim)`
            A batch of embedded sequence inputs.
        hidden : shape `(num_layers * num_directions, batch, hidden_size)`
            The initial state.
        mask : shape `(batch, max_seq_len)`
            A mask with 0 where the tokens are padding, and 1 otherwise.

        Returns :
        output : shape `(batch_size, max_seq_len, dim)`
        hidden : torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
        """

        if mask is None:
            batch_size, max_seq_len, _ = inputs.size()
            mask = inputs.new_ones((batch_size, max_seq_len), dtype=torch.uint8)

        perm_idx, sorted_seq_lens, restoration_idx = self._get_permutation_indices(mask)

        # truncated_sorted_input_embeds : (truncated_batch_size, max_seq_len, dim)
        inputs = self._perm_tensor(inputs, perm_idx, batch_dim=0)
        if hiddens is not None:
            hiddens = self._perm_tensor(hiddens, perm_idx, batch_dim=1)

        packed_embeds = pack_padded_sequence(inputs, sorted_seq_lens, batch_first=True)

        # computing!
        outputs, final_states = self.rnn(packed_embeds, hiddens)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True)  # (truncated_batch, seq_len, output_size)

        # restore outputs
        outputs = self._restore_outputs(outputs, restoration_idx, batch_dim=0)
        hiddens = self._restore_outputs(final_states, restoration_idx, batch_dim=1)

        return outputs, hiddens


def insert_zeros(tensor: torch.Tensor,
                 target_dim: int,
                 num_zeros: int) -> torch.Tensor:
    """
    Insert zeros in the `target_dim` dimension.
    """

    zeros_shape = list(tensor.shape)
    zeros_shape[target_dim] = num_zeros
    tensor = torch.cat([tensor, tensor.new_zeros(zeros_shape)], dim=target_dim)
    return tensor
