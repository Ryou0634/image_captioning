import torch

from .sequence_metric import SequenceMetric
from models.metrics.tensor_bleu import TensorBLEU


@SequenceMetric.register("bleu")
class SequenceBleu(SequenceMetric):
    def __init__(self, padding_index: int, start_index: int, end_index: int):
        self.tensor_bleu = TensorBLEU(
            ignored_indices={padding_index}, start_indices={start_index}, end_indices={end_index}
        )

    def __call__(self, predicted_tokens: torch.LongTensor, target_tokens: torch.LongTensor) -> torch.Tensor:
        sentence_bleu_list = []
        for pred, target in zip(predicted_tokens, target_tokens):
            self.tensor_bleu(pred.unsqueeze(0), target.unsqueeze(0))
            sentence_bleu = self.tensor_bleu.get_metric(reset=True)
            sentence_bleu_list.append(sentence_bleu)
        return torch.Tensor(sentence_bleu_list).to(predicted_tokens.device)
