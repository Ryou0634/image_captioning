from typing import Dict
from allennlp.training.trainer import EpochCallback, GradientDescentTrainer
from models.utils import tensor2tokens
from allennlp.nn.util import move_to_device

import logging

logger = logging.getLogger(__name__)


@EpochCallback.register("log_tokens")
class LogTokens(EpochCallback):
    def __init__(self, input_name_spaces: Dict[str, str], output_name_spaces: Dict[str, str]=None):
        self.input_name_spaces = input_name_spaces
        self.output_name_spaces = output_name_spaces

    def __call__(self, trainer: GradientDescentTrainer, epoch: int, **kwargs) -> None:
        logger.info(f"===== Sample at Epoch {epoch} =====")
        vocab = trainer.model.vocab

        # sample a instance
        batch = next(iter(trainer.data_loader))

        # log input tokens
        index = 0
        for signature, vocab_namespace in self.input_name_spaces.items():
            input_ = batch[signature]
            while isinstance(input_, dict):
                input_ = input_["tokens"]
            input_ = input_[index]
            human_redable_tokens = tensor2tokens(input_, vocab, vocab_namespace)

            logger.info(f"{signature}({vocab_namespace}): {human_redable_tokens}")

        # log output tokens
        if self.output_name_spaces:
            model = trainer.model
            model.eval()
            batch = move_to_device(batch, model.model_weight.device)
            output_dict = model(**batch)
            for signature, vocab_namespace in self.output_name_spaces.items():
                output = output_dict[signature][index]
                human_redable_tokens = tensor2tokens(output, vocab, vocab_namespace)
                logger.info(f"{signature}({vocab_namespace}): {human_redable_tokens}")

            model.get_metrics(reset=True)
