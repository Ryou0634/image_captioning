from typing import Union, Iterable
from overrides import overrides

from allennlp.training.metrics.metric import Metric


@Metric.register("meant")
class Mean(Metric):
    def __init__(self) -> None:
        self._total_value = 0.0
        self._count = 0

    @overrides
    def __call__(self, value: Union[float, Iterable]):
        """
        # Parameters

        value : `float`
            The value to average.
        """
        (value,) = self.detach_tensors(value)
        if isinstance(value, Iterable):
            self._total_value += float(sum(value))
            self._count += len(value)
        else:
            self._total_value += value
            self._count += 1

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """
        average_value = self._total_value / self._count if self._count > 0 else 0
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._total_value = 0.0
        self._count = 0
