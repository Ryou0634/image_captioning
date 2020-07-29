import torch
from allennlp.common.registrable import Registrable


class SequenceMetric(Registrable):
    def __init__(self, padding_index: int, start_index: int, end_index: int):
        self.padding_index = padding_index
        self.start_index = start_index
        self.end_index = end_index

    def __call__(self, predicted_tokens: torch.LongTensor, target_tokens: torch.LongTensor) -> torch.Tensor:
        raise NotImplementedError
