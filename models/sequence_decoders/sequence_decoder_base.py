from typing import Dict, Tuple
import torch

from allennlp.common import Registrable


class SequenceDecoder(torch.nn.Module, Registrable):
    def get_output_dim(self):
        raise NotImplementedError

    def get_input_dim(self):
        raise NotImplementedError

    def time_batch_forward(
        self, embedding_sequence: torch.Tensor, mask: torch.LongTensor = None, state: Dict[str, torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError

    def _store_step_tensor_to_state(self, state: Dict[str, torch.Tensor], tensor: torch.Tensor, key: str):
        if key not in state:
            state[key] = tensor.unsqueeze(dim=1)
        else:
            state[key] = torch.cat([state[key], tensor.unsqueeze(dim=1)], dim=1)
        return state

    def forward(
        self, embedding: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Parameters
        ============
        embedding : torch.Tensor (batch_size, embedding_size)
            A batch of embedded inputs.
        state : Dict[str, torch.Tensor]
            Typically contains context vectors from encoder, hidden states of RNN.

        Returns
        =========
        output : (batch, feature_size)
        state : Dict[str, torch.Tensor]
        """
        raise NotImplementedError
