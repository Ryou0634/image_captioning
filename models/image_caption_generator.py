from typing import Dict, List, Union

import torch

from allennlp.data import Vocabulary, TextFieldTensors
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from .image_feature_processor import ImageFeatureProcessor

from .sequence_decoders import SequenceDecoder
from .sequence_generator import SequenceGenerator
from .utils import copy_tensor_dict


@Model.register("image_caption_generator")
@SequenceGenerator.register("image_caption_generator")
class ImageCaptionGenerator(SequenceGenerator):
    def __init__(
        self,
        vocab: Vocabulary,
        image_feature_processor: ImageFeatureProcessor,
        target_embedder: TextFieldEmbedder,
        decoder: SequenceDecoder,
        tgt_vocab_namespace: str = "tgt_tokens",
        target_token_namespace: str = "tokens",
        tie_target_weights: bool = False,
        max_decoding_length: int = 100,
        beam_size: int = None,
        label_smoothing: float = 0.0,
        loss_average: str = "batch",
        attention_regularization_term: float = 0.0,
        initializer: InitializerApplicator = None,
    ):

        super().__init__(
            vocab,
            target_embedder=target_embedder,
            decoder=decoder,
            tgt_vocab_namespace=tgt_vocab_namespace,
            target_token_namespace=target_token_namespace,
            tie_target_weights=tie_target_weights,
            max_decoding_length=max_decoding_length,
            beam_size=beam_size,
            label_smoothing=label_smoothing,
            loss_average=loss_average,
        )

        self.image_feature_processor = image_feature_processor
        self.attention_regularization_term = attention_regularization_term
        self._attention_reg_loss = None

        if initializer:
            initializer(self)

    def _flatten_target_tokens(self, target_tokens: TextFieldTensors):
        target_tokens_tensor = target_tokens[self.target_token_namespace][self.target_token_namespace]
        target_tokens_tensor = target_tokens_tensor.flatten(start_dim=0, end_dim=1)
        return {self.target_token_namespace: {self.target_token_namespace: target_tokens_tensor}}

    def _expand_encoder_output(self, encoder_output: torch.Tensor, n: int):
        def repeat_tensor(tensor: torch.Tensor, n: int):
            tensor = tensor.unsqueeze(1).expand(-1, n, *(-1 for _ in range(tensor.dim() - 1)))
            tensor = tensor.flatten(start_dim=0, end_dim=1)
            return tensor

        encoder_output = {k: repeat_tensor(t, n) for k, t in encoder_output.items()}
        return encoder_output

    def forward(
        self,
        image_feature: torch.Tensor,
        target_tokens: TextFieldTensors = None,
        image_name: List[str] = None,
        start_tokens: Union[int, torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        encoder_output = self.image_feature_processor(image_feature)

        if target_tokens:
            flattened_target_tokens = self._flatten_target_tokens(target_tokens)
            num_references = target_tokens[self.target_token_namespace][self.target_token_namespace].size(1)
            expanded_encoder_output = self._expand_encoder_output(encoder_output, num_references)

        if self.training:
            output_dict = self.fit_target_tokens(flattened_target_tokens, context=expanded_encoder_output)

            if self.attention_regularization_term > 0:
                self._attention_reg_loss = self._impose_attenntion_regularization(
                    output_dict["attention_weights"], output_dict["prediction_mask"]
                )

        else:
            output_dict = encoder_output
            if target_tokens is not None:
                loss_output = self.evaluate_loss(flattened_target_tokens, context=expanded_encoder_output)
                bleu_output = self.evaluate_bleu(target_tokens, context=encoder_output)
                output_dict.update(loss_output)
                output_dict.update(bleu_output)
            else:
                output_dict.update(self.decode(start_tokens=start_tokens, context=encoder_output))

        return output_dict

    def evaluate_bleu(self, target_tokens: TextFieldTensors, context: Dict[str, torch.Tensor] = None):
        if context is not None:
            context = copy_tensor_dict(context)
        target_tokens = target_tokens[self.target_token_namespace][self.target_token_namespace]
        prediction_dict = self.decode(start_tokens=target_tokens[:, 0, 0], context=context)

        self.metrics["bleu"](prediction_dict["predicted_tokens"], target_tokens)

        return prediction_dict

    def _impose_attenntion_regularization(self, attention_weights: torch.Tensor, tgt_seq_mask: torch.LongTensor = None):
        """
        We impose a regularization for the attention weights to encourage the model to pay equal attention
        to every part of the image over the course of the generation.

        Parameters
        ----------
        attention_weights : ``torch.Tensor`` (batch_size, max_seq_length, num_context_vectors)
           Concatenation of attention weights along time direction.
        tgt_seq_mask : ``torch.LongTensor`` (batch_size, max_seq_length) or None

        Returns
        --------
        regularization_term : torch.Tensor (1, )
        """

        if tgt_seq_mask is not None:
            attention_weights = torch.einsum("btd,bt->btd", (attention_weights, tgt_seq_mask.type(torch.float)))

        attention_summed_timewise = attention_weights.sum(dim=1)  # (batch_size, num_values)
        regularization_term = (1 - attention_summed_timewise) ** 2
        return regularization_term.mean()

    def get_regularization_penalty(self) -> Union[float, torch.Tensor]:
        """
        Computes the regularization penalty for the model.
        Returns 0 if the model was not configured to use regularization.
        """
        if self._attention_reg_loss is None:
            return 0.0
        else:
            return self._attention_reg_loss

    def sampling(
        self,
        image_feature: torch.Tensor,
        start_tokens: Union[int, torch.Tensor] = None,
        context: Dict[str, torch.tensor] = None,
        batch_size: int = 1,
        **kwargs,
    ):
        encoder_output = self.image_feature_processor(image_feature)
        return self.decode_loop(start_tokens, encoder_output, batch_size, sampling_temperature=1.0)
