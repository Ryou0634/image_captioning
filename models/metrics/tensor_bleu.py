from typing import Iterable, Tuple, Dict, Set, List
from collections import Counter
import math
from overrides import overrides
import torch
from allennlp.training.metrics.metric import Metric

from .utils import padding_from_index

NgramCount = Dict[Tuple[int, ...], int]


@Metric.register("tensor_bleu")
class TensorBLEU(Metric):
    """
    Compute BLEU from tensors.
    ----------
    ngram_weights : ``Iterable[float]``, optional (default = (0.25, 0.25, 0.25, 0.25))
        Weights to assign to scores for each ngram size.
    ignored_indices : ``Set[int]``, optional (default = None)
        Indices to exclude when calculating ngrams. This should usually include
        the indices of the start, end, and pad tokens.
    start_indices : ``Set[int]``, optional (default = None)
        Indices before these tokens are replaced with padding (0).
    end_indices : ``Set[int]``, optional (default = None)
        Indices after these tokens are replaced with padding (0).

    This implementation is based on allennlp, but extented to a reference set of size more than 1.
    """

    def __init__(self,
                 ngram_weights: Iterable[float] = (0.25, 0.25, 0.25, 0.25),
                 ignored_indices: Set[int] = None,
                 start_indices: Set[int] = None,
                 end_indices: Set[int] = None) -> None:
        self._ngram_weights = ngram_weights
        self._ignored_indices = ignored_indices or set()
        self._start_indices = start_indices or set()
        self._end_indices = end_indices or set()
        self._precision_matches: Dict[int, int] = Counter()
        self._precision_totals: Dict[int, int] = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0

    @overrides
    def __call__(self,  # type: ignore
                 predictions: torch.LongTensor,
                 gold_targets: torch.LongTensor) -> None:
        """
        Update precision counts.
        Parameters
        ----------
        predictions : ``torch.LongTensor``, required
            Batched predicted tokens of shape `(batch_size, max_sequence_length)`.
        gold_targets : ``torch.LongTensor``, required
            Batched reference (gold) translations
            with shape `(batch_size, max_gold_sequence_length)` or `(batch_size, num_references, max_gold_sequence_length)`.
        Returns
        -------
        None
        """
        predictions, gold_targets = self.detach_tensors(predictions, gold_targets)

        # remove unwanted tokens
        for idx in self._start_indices:
            padding_idx = next(iter(self._ignored_indices))
            predictions = padding_from_index(predictions, idx, padding_index=padding_idx, pad_before=True)
            gold_targets = padding_from_index(gold_targets, idx, padding_index=padding_idx, pad_before=True)

        for idx in self._end_indices:
            padding_idx = next(iter(self._ignored_indices))
            predictions = padding_from_index(predictions, idx, padding_index=padding_idx)
            gold_targets = padding_from_index(gold_targets, idx, padding_index=padding_idx)

        if len(gold_targets.size()) == 2:
            gold_targets = gold_targets.unsqueeze(1)

        for ngram_size, _ in enumerate(self._ngram_weights, start=1):
            precision_matches, precision_totals = self._get_modified_precision_counts(
                predictions, gold_targets, ngram_size)
            self._precision_matches[ngram_size] += precision_matches
            self._precision_totals[ngram_size] += precision_totals
        if not self._ignored_indices:
            self._prediction_lengths += predictions.size(0) * predictions.size(1)
            self._reference_lengths += gold_targets.size(0) * gold_targets.size(2)
        else:
            valid_predictions_mask = self._get_valid_tokens_mask(predictions)
            valid_gold_targets_mask = self._get_valid_tokens_mask(gold_targets)

            best_match_reference_lengths = self._get_best_match_ref_lengths(
                prediction_lengths=valid_predictions_mask.sum(dim=1),
                references_lengths=valid_gold_targets_mask.sum(dim=2))

            self._prediction_lengths += valid_predictions_mask.sum().item()
            self._reference_lengths += best_match_reference_lengths.sum().item()

    @overrides
    def reset(self) -> None:
        self._precision_matches = Counter()
        self._precision_totals = Counter()
        self._prediction_lengths = 0
        self._reference_lengths = 0

    def _count_ngrams(self,
                      tensor: torch.LongTensor,
                      ngram_size: int) -> NgramCount:
        ngram_counts: Dict[Tuple[int, ...], int] = Counter()
        if ngram_size > tensor.size(-1):
            return ngram_counts
        for start_position in range(ngram_size):
            for tensor_slice in tensor[start_position:].split(ngram_size, dim=-1):
                if tensor_slice.size(-1) < ngram_size:
                    break
                ngram = tuple(x.item() for x in tensor_slice)
                if any(x in self._ignored_indices for x in ngram):
                    continue
                ngram_counts[ngram] += 1
        return ngram_counts

    @staticmethod
    def _aggregate_ngram_counters(ngram_counts_list: List[NgramCount]) -> NgramCount:
        aggregated_counts = Counter()

        # get all n_grams
        n_grams = set()
        for counter in ngram_counts_list:
            n_grams = n_grams.union(counter.keys())

        # aggregate all counters
        for n_gram in n_grams:
            n_gram_freq_list = [counter[n_gram] for counter in ngram_counts_list]
            aggregated_counts[n_gram] = max(n_gram_freq_list)

        return aggregated_counts

    def _get_modified_precision_counts(self,
                                       predicted_tokens: torch.LongTensor,
                                       references_tokens: torch.LongTensor,
                                       ngram_size: int) -> Tuple[int, int]:
        """
        Compare the predicted tokens to the reference (gold) tokens at the desired
        ngram size and calculate the numerator and denominator for a modified
        form of precision.
        The numerator is the number of ngrams in the predicted sentences that match
        with an ngram in the corresponding reference sentence, clipped by the total
        count of that ngram in the reference sentence. The denominator is just
        the total count of predicted ngrams.
        """
        clipped_matches = 0
        total_predicted = 0
        for batch_num in range(predicted_tokens.size(0)):

            # get n_gram counts for predicted_tokens
            predicted_row = predicted_tokens[batch_num, :]
            predicted_ngram_counts = self._count_ngrams(predicted_row, ngram_size)

            # get clipped n_gram counts for references_tokens
            references_row = references_tokens[batch_num, :]
            reference_ngram_counts_list = []
            for ref_num in range(references_row.size(0)):
                reference_ngram_counts_list.append(self._count_ngrams(references_row[ref_num, :], ngram_size))
            reference_ngram_counts = self._aggregate_ngram_counters(reference_ngram_counts_list)

            # get clipped_matches
            for ngram, count in predicted_ngram_counts.items():
                clipped_matches += min(count, reference_ngram_counts[ngram])
                total_predicted += count
        return clipped_matches, total_predicted

    def _get_brevity_penalty(self) -> float:
        if self._prediction_lengths > self._reference_lengths:
            return 1.0
        if self._reference_lengths == 0 or self._prediction_lengths == 0:
            return 0.0
        return math.exp(1.0 - self._reference_lengths / self._prediction_lengths)

    def _get_valid_tokens_mask(self, tensor: torch.LongTensor) -> torch.ByteTensor:
        valid_tokens_mask = tensor.new_ones(tensor.size(), dtype=torch.uint8)
        for index in self._ignored_indices:
            valid_tokens_mask = valid_tokens_mask & (tensor != index).byte()
        return valid_tokens_mask

    @staticmethod
    def _get_best_match_ref_lengths(prediction_lengths: torch.LongTensor,
                                    references_lengths: torch.LongTensor) -> torch.LongTensor:
        length_distance = abs(prediction_lengths[:, None] - references_lengths)
        best_match_length_indices = length_distance.argmin(dim=1)
        best_match_lengths = references_lengths[range(len(best_match_length_indices)), best_match_length_indices]
        return best_match_lengths

    @overrides
    def get_metric(self, reset: bool = False) -> float:
        brevity_penalty = self._get_brevity_penalty()

        ngram_scores = (weight * (math.log(self._precision_matches[n] + 1e-13) -
                                  math.log(self._precision_totals[n] + 1e-13))
                        for n, weight in enumerate(self._ngram_weights, start=1))
        bleu = brevity_penalty * math.exp(sum(ngram_scores)) * 100
        if reset:
            self.reset()
        return bleu
