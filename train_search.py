from typing import Dict, Any, List
import _jsonnet
import json
import itertools
import click
from pathlib import Path
from multiprocessing import Queue, Process


from allennlp.common.params import Params

from allennlp.common.util import import_module_and_submodules
from allennlp.commands.train import train_model

from train import RESULT_DIR

import logging

logging.basicConfig(format="%(asctime)s %(levelname)s:%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_parameter_json(parameters_path: str):
    parameters = json.loads(_jsonnet.evaluate_file(parameters_path))
    keys = parameters.keys()
    values = [parameters[k] for k in keys]
    for combinations in itertools.product(*values):
        yield dict(zip(keys, combinations))


def override_config(config: Dict, parameters: Dict[str, Any]):
    def replace_with_keys(dictionary: Dict, keys: List, value: Any):
        k = keys.pop(0)
        if len(keys) == 0:
            dictionary[k] = value
        else:
            replace_with_keys(dictionary[k], keys, value)

    for keys, value in parameters.items():
        keys = keys.split(".")
        replace_with_keys(config, keys, value)
    return config


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.argument("parameters_path", type=click.Path(exists=True))
@click.option("--overrides", type=str)
@click.option("--cuda", type=str, default=["-1"], multiple=True)
@click.option("--seed", type=int, default=0)
@click.option("--include-package", type=str, default="models,experiments")
@click.option("--force", is_flag=True)
def parameter_search(
    config_path: str, parameters_path: str, overrides: str, cuda: str, seed: int, include_package: str, force: bool,
):
    include_package = include_package.split(",")
    for package in include_package:
        import_module_and_submodules(package)

    params = Params.from_file(config_path, overrides)

    # set random seed
    params["numpy_seed"] = seed
    params["pytorch_seed"] = seed
    params["random_seed"] = seed

    # prepare the serialization directory
    config_path = Path(config_path)
    experiment_name = config_path.parent.stem
    setting_name = config_path.stem

    task_q = Queue()
    for i, hyper_params in enumerate(parse_parameter_json(parameters_path)):
        serialization_dir = RESULT_DIR / experiment_name / setting_name / f"search_{i}"

        duplicated_params = params.duplicate()
        duplicated_params = override_config(duplicated_params, hyper_params)

        task_q.put({"params": duplicated_params, "serialization_dir": str(serialization_dir)})

    logger.info(f"Preform search with {task_q.qsize()} settings...")

    def consume_tasks_on_device(task_q: Queue, cuda: List[int]):
        while not task_q.empty():
            task = task_q.get()
            params = task["params"]

            # set GPU devices
            if len(cuda) > 1:
                params["distributed"] = {"cuda_devices": cuda}
            else:
                params["trainer"]["cuda_device"] = cuda[0]

            best_model = train_model(
                params,
                task["serialization_dir"],
                file_friendly_logging=True,
                include_package=include_package,
                force=force,
            )

    cuda_list = [[int(c) for c in device_ids.split(",")] for device_ids in cuda]
    logger.info(f"cuda_list: {cuda_list}")
    for cuda in cuda_list:
        Process(target=consume_tasks_on_device, args=(task_q, cuda)).start()


if __name__ == "__main__":
    parameter_search()
