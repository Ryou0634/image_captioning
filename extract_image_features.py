from typing import Tuple

import click
import glob
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import torchvision.models as models

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


class ImageFeatureExtractor(nn.Module):
    """
    Extract visual features using a pretrained model.
    """

    def __init__(self, model_name: str = "vgg16", num_layers: int = 31):
        """
        :param model_name:
        :param num_layers: show and tell -> 31, show, attend and tell -> 24
        """
        super().__init__()
        pretrained_model = getattr(models, model_name)(pretrained=True)
        feature_extractor = dict(pretrained_model.named_children())["features"]
        extracted = list(feature_extractor.children())[:num_layers]
        self.model = torch.nn.Sequential(*extracted)

        self.model.eval()

    def forward(self, image_tensor: torch.Tensor):
        return self.model(image_tensor)


class ImageReader:
    def __init__(self, image_tensor_size: Tuple[int, int] = (224, 224)):
        self.image_tensor_size = image_tensor_size

        self.tensorize = transforms.Compose(
            [
                transforms.Resize(self.image_tensor_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def read_image(self, image_path: str):
        img = Image.open(image_path)
        img = img.convert("RGB")
        return self.tensorize(img)


@click.command()
@click.argument("image_directory", type=click.Path(exists=True))
@click.argument("output_directory", type=click.Path())
@click.option("--pretrained-model", type=str, default="vgg16")
@click.option("--pretrained-num-layers", type=int, default=24)
@click.option("--device", type=str, default="cpu")
@click.option("--batch-size", type=int, default=32)
@click.option("--ext", type=str, default="jpg")
def extract_image_features(
    image_directory: str,
    output_directory: str,
    pretrained_model: str,
    pretrained_num_layers: int,
    device: str,
    batch_size: int,
    ext: str,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    device = torch.device(device)
    reader = ImageReader()
    image_path_list = sorted(glob.glob(str(Path(image_directory) / f"*.{ext}")))
    logger.info(f"Reading {len(image_path_list)} images in total...")
    image_tensors = [reader.read_image(path) for path in image_path_list]
    data_loader = DataLoader(image_tensors, batch_size=batch_size, shuffle=False)

    logger.info(f"Loading the pretrained model {pretrained_model}...")
    feature_extractor = ImageFeatureExtractor(pretrained_model, pretrained_num_layers).to(device)

    model_outputs = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            output_tensor = feature_extractor(batch)
            if idx == 0:
                logger.info(f"The size of extracted feature: {output_tensor.size()[1:]}")
            model_outputs.append(output_tensor.detach().cpu())
    image_filenames = [Path(path).name for path in image_path_list]
    with open(output_directory / f"image_filenames_{pretrained_model}_layer{pretrained_num_layers}.txt", "w") as f:
        f.write("\n".join(image_filenames))

    model_outputs = torch.cat(model_outputs, dim=0).numpy()
    np.save(output_directory / f"{pretrained_model}_layer{pretrained_num_layers}.npy", model_outputs)


if __name__ == "__main__":
    extract_image_features()
