import torch
import torch.nn as nn
from .image_feature_processor import ImageFeatureProcessor


@ImageFeatureProcessor.register("sequence")
class SequenceFeature(ImageFeatureProcessor):
    def __init__(
        self, input_size: int, output_size: int, use_hidden: bool = True, use_cell: bool = True, dropout: float = 0.1
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.annotation_linear = nn.Linear(input_size, output_size)

        self.use_hidden = use_hidden
        self.use_cell = use_cell
        if use_hidden:
            self.hiddens_linear = nn.Linear(output_size, output_size)
        if use_cell:
            self.cells_linear = nn.Linear(output_size, output_size)

        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.Tanh()

    def forward(self, image: torch.Tensor):
        image = image.flatten(start_dim=2).transpose(1, 2)  # shape: (batch_size, num_vectors, feature_size)
        annotation_vectors = self.activation(self.annotation_linear(image))
        output_dict = {"encoder_output": annotation_vectors}

        if self.use_hidden:
            annotation_mean_vector = annotation_vectors.sum(dim=1) / annotation_vectors.size(0)
            output_dict["hiddens"] = self.dropout(self.activation(self.hiddens_linear(annotation_mean_vector)))

        if self.use_cell:
            annotation_mean_vector = annotation_vectors.sum(dim=1) / annotation_vectors.size(0)
            output_dict["cells"] = self.dropout(self.activation(self.cells_linear(annotation_mean_vector)))

        return output_dict
