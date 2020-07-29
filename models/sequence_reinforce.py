import torch
import torch.nn.functional as F

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from models.metrics.mean import Mean
from models.metrics.standard_deviation import StandardDeviation

from .sequence_generator import SequenceGenerator
from .utils import get_target_mask
from .sequence_metrics.sequence_bleu import SequenceBleu


@Model.register("sequence_reinforce")
class SequenceReinforce(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        sequence_generator: SequenceGenerator,
        self_critic: bool = False,
        initializer: InitializerApplicator = None,
    ):
        super().__init__(vocab)
        self.sequence_generator = sequence_generator
        self.reward_function = SequenceBleu(
            padding_index=0,
            start_index=self.sequence_generator.target_start_index,
            end_index=self.sequence_generator.target_end_index,
        )
        self.self_critic = self_critic

        self.target_token_namespace = self.sequence_generator.target_token_namespace

        self.metrics = {
            "average_raw_reward": Mean(),
            "average_baselined_reward": Mean(),
            "raw_reward_variance": StandardDeviation(),
            "baselined_reward_variance": StandardDeviation(),
        }

        if initializer is not None:
            initializer(self.sequence_generator)

    @property
    def model_weight(self):
        return self.sequence_generator.model_weight

    def forward(self, target_tokens: TextFieldTensors, **kwargs):

        if self.training:
            target_tokens = target_tokens[self.target_token_namespace][self.target_token_namespace]
            batch_size = target_tokens.size(0)
            sampling_output = self.sequence_generator.sampling(batch_size=batch_size, **kwargs)
            logits = sampling_output["logits"]
            predicted_tokens = sampling_output["predicted_tokens"]  # shape: (batch_size, sequence_length)
            log_probs = F.log_softmax(logits, dim=2)  # shape: (batch_size, sequence_length, vocab_size)

            # shape: (batch_size, sequence_length)
            log_prob_for_predicted_tokens = log_probs.gather(dim=2, index=predicted_tokens.unsqueeze(2)).squeeze(2)

            mask = get_target_mask(predicted_tokens, end_index=self.sequence_generator.target_end_index)
            log_prob_for_predicted_tokens *= mask

            # shape: (batch_size, )
            log_prob_per_sequence = log_prob_for_predicted_tokens.sum(dim=1)

            # shape: (batch_size, )
            reward = self.reward_function(predicted_tokens, target_tokens)
            self.metrics["average_raw_reward"](reward)
            self.metrics["raw_reward_variance"](reward)

            if self.self_critic:
                self.sequence_generator.eval()
                start_tokens = target_tokens.new_full(
                    size=(batch_size,), fill_value=self.sequence_generator.target_start_index, dtype=torch.long
                )
                greedy_output = self.sequence_generator(start_tokens=start_tokens, **kwargs)
                self.sequence_generator.train()
                baseline_reward = self.reward_function(greedy_output["predicted_tokens"], target_tokens)
            else:
                baseline_reward = self.metrics["average_reward"].get_metric(reset=False)

            reward -= baseline_reward
            self.metrics["average_baselined_reward"](reward)
            self.metrics["baselined_reward_variance"](reward)

            loss = -(reward * log_prob_per_sequence).sum()

            return {"loss": loss, "predicted_tokens": predicted_tokens}
        else:
            return self.sequence_generator(target_tokens=target_tokens, **kwargs)

    def get_metrics(self, reset: bool = False):
        metrics = {
            **self.sequence_generator.get_metrics(reset=reset),
            **{k: v.get_metric(reset=reset) for k, v in self.metrics.items()},
        }
        return metrics
