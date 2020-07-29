import argparse
from pathlib import Path
import os
import logging
import click

from allennlp.common.params import Params
from allennlp.commands.train import train_model
from allennlp.commands.evaluate import evaluate_from_args
from allennlp.common.util import import_module_and_submodules

EXPERIMENT_DIR = Path("experiments")
RESULT_DIR = Path("results")


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--cuda", type=str, default="-1")
@click.option("--seed", type=int, default=0)
@click.option("--overrides", type=str)
@click.option("--include-package", type=str, default="models,experiments")
@click.option("--dry-run", is_flag=True)
@click.option("--force", is_flag=True)
@click.option("--do-not-save-model", is_flag=True)
@click.option("--save-name", type=str, default="default")
def train(
    config_path: str,
    cuda: str,
    seed: int,
    overrides: str,
    include_package: str,
    dry_run: bool,
    force: bool,
    do_not_save_model: bool,
    save_name: str
):
    params = Params.from_file(config_path, overrides)

    # set GPU devices
    cuda = [int(c) for c in cuda.split(",")]
    if len(cuda) > 1:
        params["distributed"] = {"cuda_devices": cuda}
    else:
        params["trainer"]["cuda_device"] = cuda[0]

    # set random seed
    params["numpy_seed"] = seed
    params["pytorch_seed"] = seed
    params["random_seed"] = seed

    # prepare the serialization directory
    config_path = Path(config_path)
    experiment_name = config_path.parent.stem
    setting_name = config_path.stem
    serialization_dir = RESULT_DIR / experiment_name / setting_name / save_name

    include_package = include_package.split(",")
    for package in include_package:
        import_module_and_submodules(package)

    evaluate_on_test = "test_data_path" in params

    if evaluate_on_test:
        test_data_path = params["test_data_path"]
    best_model = train_model(
        params,
        str(serialization_dir),
        file_friendly_logging=True,
        force=force,
        include_package=include_package,
        dry_run=dry_run,
    )
    if evaluate_on_test and not dry_run:
        evaluate_args = argparse.Namespace(
            archive_file=str(serialization_dir / "model.tar.gz"),
            input_file=test_data_path,
            output_file=str(serialization_dir / "metrics_test.json"),
            cuda_device=cuda[0],
            overrides=overrides,
            weights_file=None,
            extend_vocab=None,
            embedding_sources_mapping=None,
            batch_weight_key=None,
            batch_size=None,
        )
        evaluate_from_args(evaluate_args)

    if do_not_save_model:
        os.remove(serialization_dir / "model.tar.gz")
        os.remove(serialization_dir / "best.th")


if __name__ == "__main__":
    train()
