import torch
import torch.nn as nn


class LockedDropout(nn.Module):
    """
    Apply the dropout mask for the certain dimension.
    The typical use case is when you want to apply the same dropout
    along the time-step dimension in RNN.
    """

    def __init__(self, locked_dim: int = -1, p: float = 0.):
        super().__init__()
        self._locked_dim = locked_dim
        assert 0 <= p <= 1.0
        self._p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.training or not self._p:
            return x

        tensor_size = list(x.size())
        if self._locked_dim > -1:
            tensor_size[self._locked_dim] = 1

        mask = x.new_empty(tensor_size).bernoulli_(1 - self._p).requires_grad_(False)

        if self._p != 1.0:
            mask = mask / (1 - self._p)
        mask = mask.expand_as(x)

        return mask * x
