import click
import argparse
from pathlib import Path

from allennlp.common.params import Params
from allennlp.commands.evaluate import evaluate_from_args
from allennlp.common.util import import_module_and_submodules

import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("serialization_dir", type=click.Path(exists=True))
@click.option("--data_path", type=click.Path(exists=True))
@click.option("--cuda", type=int, default=-1)
@click.option("--overrides", type=str)
@click.option("--include-package", type=str, default="models,experiments")
def evaluate(serialization_dir: str, data_path: str, cuda: int, overrides: str, include_package: str):
    include_package = include_package.split(",")
    for package in include_package:
        import_module_and_submodules(package)

    serialization_dir = Path(serialization_dir)

    config_path = serialization_dir / "config.json"
    params = Params.from_file(config_path, overrides)

    data_path = data_path or params["test_data_path"]

    evaluate_args = argparse.Namespace(
        archive_file=str(serialization_dir / "model.tar.gz"),
        input_file=data_path,
        output_file=str(serialization_dir / "metrics_test.json"),
        cuda_device=cuda,
        overrides=overrides,
        weights_file=None,
        extend_vocab=None,
        embedding_sources_mapping=None,
        batch_weight_key=None,
        batch_size=None,
    )
    evaluate_from_args(evaluate_args)


if __name__ == "__main__":
    evaluate()
