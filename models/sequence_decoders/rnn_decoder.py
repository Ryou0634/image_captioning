from typing import Union, List, Dict, Tuple

import torch

from models.submodules.stacked_rnn import StackedRNN
from .sequence_decoder_base import SequenceDecoder


@SequenceDecoder.register("rnn_decoder")
class RNNDecoder(SequenceDecoder):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        residual: Union[bool, List[bool]] = False,
        inter_layer_dropout: float = 0.1,
        weight_dropout: float = 0.0,
        rnn: str = "LSTM",
    ):
        super().__init__()
        self._input_size = input_size
        self._hidden_size = hidden_size
        self.num_layers = num_layers

        assert rnn in ["RNN", "LSTM", "GRU"], f"{rnn} is currently not supported."
        self.rnn = StackedRNN(
            input_size,
            hidden_size,
            rnn_type=rnn,
            batch_first=True,
            bidirectional=False,
            num_layers=num_layers,
            weight_dropout=weight_dropout,
            inter_layer_dropout=inter_layer_dropout,
            residual=residual,
        )
        self.rnn_type = rnn

    def get_output_dim(self):
        return self._hidden_size

    def get_input_dim(self):
        return self._input_size

    def time_batch_forward(
        self, embedding_sequence: torch.Tensor, mask: torch.LongTensor = None, state: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        #  (batch, seq_len, embed_size)
        n_decoding_steps = embedding_sequence.size(1)
        outputs = []
        for t in range(n_decoding_steps):
            rnn_output, state = self.forward(embedding_sequence[:, t], state)
            outputs.append(rnn_output)
        return torch.stack(outputs, dim=1), state

    def _expand_hiddens_if_needed(self, state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if "hiddens" in state:
            if state["hiddens"].dim() == 2:
                state["hiddens"] = state["hiddens"].unsqueeze(1).expand(-1, self.num_layers, -1)
        if "cells" in state:
            if state["cells"].dim() == 2:
                state["cells"] = state["cells"].unsqueeze(1).expand(-1, self.num_layers, -1)
        return state

    def forward(
        self, embedding: torch.Tensor, state: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        state = state or {}
        state = self._expand_hiddens_if_needed(state)

        # tensors in ``state`` are all batch-first, so permute it to feed in rnn_unit of PyTorch.
        batch_size = embedding.size(0)

        if "hiddens" in state:
            hiddens_and_maybe_cells = state["hiddens"].permute(1, 0, 2).contiguous()
        else:
            hiddens_and_maybe_cells = embedding.new_zeros((self.num_layers, batch_size, self._hidden_size))

        if self.rnn_type == "LSTM":
            if "cells" in state:
                hiddens_and_maybe_cells = (hiddens_and_maybe_cells, state["cells"].permute(1, 0, 2).contiguous())
            else:
                hiddens_and_maybe_cells = (
                    hiddens_and_maybe_cells,
                    embedding.new_zeros((self.num_layers, batch_size, self._hidden_size)),
                )

        tensors, hiddens_and_maybe_cells = self.rnn(embedding.unsqueeze(1), hiddens_and_maybe_cells)

        if self.rnn_type == "LSTM":
            state["hiddens"] = hiddens_and_maybe_cells[0].permute(1, 0, 2).contiguous()
            state["cells"] = hiddens_and_maybe_cells[1].permute(1, 0, 2).contiguous()
        else:
            state["hiddens"] = hiddens_and_maybe_cells.permute(1, 0, 2).contiguous()

        # (batch, num_directions * hidden_size)
        tensors = tensors.squeeze(1)
        return tensors, state
