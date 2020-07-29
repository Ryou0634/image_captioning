from typing import List
import torch


def get_max_len_of_nested_sequence(lis: List):
    if isinstance(lis[0], list):
        return max([get_max_len_of_nested_sequence(l) for l in lis])
    else:
        return len(lis)


def pad_nested_sequence_to_length(lis: List, pad_length: int, padding: int = 0):
    if isinstance(lis[0][0], list):
        return [pad_nested_sequence_to_length(l, pad_length) for l in lis]
    else:
        for i, l in enumerate(lis):
            lis[i] += [padding for _ in range(pad_length - len(l))]
        return lis


def padding_from_index(tensor: torch.Tensor,
                       target_index: int,
                       padding_index: int = 0,
                       pad_before: bool = False,
                       recursing: bool = False) -> torch.Tensor:
    """
    Replace token ids after the end token with the padding tokens.

    Example:
        tensor = torch.LongTensor([[1, 2, 3, 4, 5],
                                   [2, 3, 4, 5, 6]])
        end_index = 4
        padding_index = 0

        padding_from_index(tensor, end_index, padding_index)
        -> torch.LongTensor([[1, 2, 3, 4, 0],
                             [2, 3, 4, 0, 0]])
    """

    if not recursing:
        tensor = tensor.detach().clone()

    if tensor.dim() == 1:
        tensor_list = list(tensor)
        if target_index in tensor_list:
            target_i = tensor_list.index(target_index)
            if pad_before:
                tensor[:target_i + 1] = padding_index
            else:
                tensor[target_i:] = padding_index
    else:
        for row in tensor:
            padding_from_index(row, target_index, padding_index, pad_before, recursing=True)

    return tensor
