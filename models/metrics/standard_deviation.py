from typing import Union, Iterable
from overrides import overrides
import numpy as np

from allennlp.training.metrics.metric import Metric


@Metric.register("standard_deviation")
class StandardDeviation(Metric):

    def __init__(self) -> None:
        self._values = []

    @overrides
    def __call__(self, value: Union[float, Iterable]):
        """
        # Parameters

        value : `float`
            The value to average.
        """
        (value,) = self.detach_tensors(value)
        if isinstance(value, Iterable):
            self._values += [float(v) for v in value]
        else:
            self._values.append(value)

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """
        average_value = np.std(self._values)
        if reset:
            self.reset()
        return average_value

    @overrides
    def reset(self):
        self._values = []

