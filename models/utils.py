from typing import Dict, List, Union
import numpy as np
import torch
import logging
from enum import Enum, auto

from allennlp.data.vocabulary import Vocabulary

PADDING_INDEX = 0

logger = logging.getLogger(__name__)


def get_target_mask(tensor: torch.Tensor,
                    end_index: int,
                    recursing: bool = False) -> torch.Tensor:
    """
    Mask tokens after the end_index.
    Note that this function leave the end index unmasked.
    """

    # Firstly, clone the tensor and fill this with 1 or 0 in place
    if not recursing:
        tensor = tensor.detach().clone()

    if tensor.dim() == 1:
        tensor_list = list(tensor)
        if end_index in tensor_list:
            target_i = tensor_list.index(end_index)
            target_i = min(target_i + 1, len(tensor_list))
            tensor[target_i:] = 0
            tensor[:target_i] = 1
        else:
            tensor[:] = 1
    else:
        for row in tensor:
            get_target_mask(row, end_index, recursing=True)

    return tensor


def tensor_to_string_tokens(token_tensor: torch.Tensor,
                            vocab: Vocabulary,
                            namespace: str,
                            end_index: int) -> List[List[str]]:
    """

    Parameters
    ----------
    token_tensor : torch.Tensor (batch_size, sequence_length)
    vocab : Vocabulary
    namespace : str
    end_index : int
    """
    tensor_mask = get_target_mask(token_tensor, end_index=end_index)

    predicted_indices = token_tensor.detach().cpu().numpy()
    tensor_mask = tensor_mask.detach().cpu().numpy()
    predicted_indices *= tensor_mask

    all_predicted_tokens: List[List[str]] = []
    for indices in predicted_indices:
        predicted_tokens = [vocab.get_token_from_index(idx, namespace=namespace)
                            for idx in indices if (idx != 0 and idx != end_index)]
        all_predicted_tokens.append(predicted_tokens)
    return all_predicted_tokens


def tensor_to_string_tokens_3d(token_tensor: torch.Tensor,
                               vocab: Vocabulary,
                               namespace: str,
                               end_index: int):
    all_predicted_tokens: List[List[str]] = []
    for sentence_tensor in token_tensor:
        characters_list = tensor_to_string_tokens(sentence_tensor, vocab, namespace, end_index)
        words = [''.join(chars) for chars in characters_list]
        all_predicted_tokens.append(words)
    return all_predicted_tokens


class TrainingState(Enum):
    INIT = auto()
    TRAINING = auto()
    VALIDATION = auto()


class TrainingStateChangeDetector:
    def __init__(self):
        self.current_state = TrainingState.INIT

    def state_has_changed(self, new_state_is_training: bool):
        new_state = TrainingState.TRAINING if new_state_is_training else TrainingState.VALIDATION

        prev_state = self.current_state
        self.current_state = new_state

        return self.current_state != prev_state


def log_internal_tokens(current_state: str,
                        vocab: Vocabulary,
                        sequence_tensors: List[torch.Tensor],
                        vocab_name_spaces: List[str],
                        display_names: List[str],
                        index: int = None):
    logger.info(f"===== Sample from {current_state} instances =====")

    batch_size = len(sequence_tensors[0])
    index = index or np.random.randint(batch_size)  # randomly select one instance in the batch

    for data, name_space, display_name in zip(sequence_tensors, vocab_name_spaces, display_names):
        if name_space is not None:
            sampled_data = tensor2tokens(data[index], vocab, name_space)
        else:
            sampled_data = data[index]
        logger.info(f"{display_name} {name_space}: {sampled_data}")


def tensor2tokens(index_sequence: torch.Tensor,  # shape : (seq_len, )
                  vocab: Vocabulary,
                  name_space: str) -> List[Union[str, List[str]]]:
    if index_sequence.dim() == 1:
        tokens = [vocab.get_token_from_index(idx.item(), name_space) for idx in index_sequence]
    elif index_sequence.dim() == 2:
        tokens = [[vocab.get_token_from_index(idx.item(), name_space) for idx in idx_list]
                  for idx_list in index_sequence]
    return tokens


def transfer_output_dict(final_output: Dict[str, torch.tensor],
                         module_output_dict: Dict[str, torch.tensor],
                         label: str):
    for key, item in module_output_dict.items():
        final_output[f"{label}/{key}"] = item
    return final_output


def append_prefix_to_metrics_dict(metrics_dict: Dict[str, float], prefix: str) -> Dict[str, float]:
    output_metrics = {}
    for metric_name, metric_value in metrics_dict.items():
        module_metric_name = prefix + '/' + metric_name
        output_metrics[module_metric_name] = metric_value
    return output_metrics


def copy_tensor_dict(tensor_dict: Dict[str, torch.tensor]) -> Dict[str, torch.Tensor]:
    copied = {}
    for key, tensor in tensor_dict.items():
        copied[key] = tensor.clone()
    return copied
