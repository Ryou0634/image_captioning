from typing import Dict
import torch
from allennlp.common.registrable import Registrable


class ImageFeatureProcessor(torch.nn.Module, Registrable):
    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        image : torch.Tensor (batch_size, channel, width, height)
        """
        raise NotImplementedError
