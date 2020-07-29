from typing import Tuple, List, Iterable, Dict
import os
import logging
from pathlib import Path

import torch

import allennlp
from allennlp.models.archival import load_archive
from allennlp.common import Params
from allennlp.data import Vocabulary, DatasetReader, Instance, DataLoader
from allennlp.data.dataloader import DataLoader
from allennlp.models.archival import CONFIG_NAME
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.common.util import import_module_and_submodules, prepare_environment


def instantiate_model_from_config(
    config_file_path: str, cuda_device: int = -1, overrides: str = None, include_package: str = "models"
) -> Model:
    logging.disable(logging.INFO)
    import_module_and_submodules(include_package)

    params = Params.from_file(config_file_path, overrides)

    vocab_dir = params.pop("vocabulary").pop("directory_path")
    vocab = Vocabulary.from_files(vocab_dir)

    model = Model.from_params(vocab=vocab, params=params.pop("model"))

    if cuda_device >= 0:
        model.cuda(cuda_device)
    else:
        model.cpu()

    return model


def load_dataset_reader_from_config(
    config_file_path: str, include_package: str = "models", reader_name: str = "dataset_reader", overrides: str = None
):
    logging.disable(logging.INFO)
    import_module_and_submodules(include_package)
    params = Params.from_file(config_file_path, overrides)
    dataset_reader = DatasetReader.from_params(params.pop(reader_name))
    return dataset_reader


def load_model_and_dataset_reader(
    arcive_file: str, include_package: str = "models", cuda_device: int = -1, overrides: str = None
) -> Tuple[Model, DatasetReader]:
    logging.disable(logging.INFO)
    import_module_and_submodules(include_package)
    archive = load_archive(arcive_file, cuda_device=cuda_device, overrides=overrides)
    config = archive.config
    prepare_environment(config)

    model = archive.model
    model.eval()

    dataset_reader = DatasetReader.from_params(config.pop("dataset_reader"))
    return model, dataset_reader


def load_model_from_file(
    serialization_dir: str,
    weights_file: str = _DEFAULT_WEIGHTS,
    include_package: str = "models",
    cuda_device: int = -1,
    overrides: str = None,
):
    logging.disable(logging.INFO)
    config = Params.from_file(os.path.join(serialization_dir, CONFIG_NAME), overrides)

    import_module_and_submodules(include_package)

    model = Model.load(
        config,
        weights_file=os.path.join(serialization_dir, weights_file),
        serialization_dir=serialization_dir,
        cuda_device=cuda_device,
    )
    return model



