import torch
import torch.nn as nn
from .image_feature_processor import ImageFeatureProcessor


@ImageFeatureProcessor.register("fixed_length")
class FixedLengthFeature(ImageFeatureProcessor):
    def __init__(self, input_size: int, output_size: int, for_lstm: bool = True, dropout: float = 0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.for_lstm = for_lstm
        self.hiddens_linear = nn.Linear(input_size, output_size)
        if for_lstm:
            self.cells_linear = nn.Linear(input_size, output_size)

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Tanh()

    def forward(self, image: torch.Tensor):
        features = image.flatten(start_dim=1)
        output_dict = {
            "hiddens": self.dropout(self.activation(self.hiddens_linear(features)))
        }
        if self.for_lstm:
            output_dict["cells"] = self.dropout(self.activation(self.cells_linear(features)))
        return output_dict
