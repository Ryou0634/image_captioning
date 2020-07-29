from typing import Collection
from overrides import overrides
import torch
from torch.nn.functional import embedding

from allennlp.common import Params
from allennlp.data import Vocabulary
from allennlp.modules.time_distributed import TimeDistributed
from allennlp.nn import util

from allennlp.modules import TokenEmbedder
from allennlp.modules.token_embedders.embedding import Embedding, _read_pretrained_embeddings_file

from .locked_dropout import LockedDropout


def dropout_mask(x: torch.Tensor, size: Collection[int], p: float):
    """Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."""
    return x.new_empty(size).bernoulli_(1 - p).div_(1 - p)


@TokenEmbedder.register("my_embedding")
class EmbeddingLayer(Embedding):
    """Add word dropout to the allennlp implementation."""

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 projection_dim: int = None,
                 weight: torch.FloatTensor = None,
                 padding_index: int = None,
                 trainable: bool = True,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 dropout: float = 0.0,
                 word_dropout: float = 0.) -> None:
        super().__init__(num_embeddings=num_embeddings,
                         embedding_dim=embedding_dim,
                         projection_dim=projection_dim,
                         weight=weight,
                         padding_index=padding_index,
                         trainable=trainable,
                         max_norm=max_norm,
                         norm_type=norm_type,
                         scale_grad_by_freq=scale_grad_by_freq,
                         sparse=sparse)

        self.word_dropout = word_dropout
        self.dropout = LockedDropout(locked_dim=1, p=dropout)

    @overrides
    def forward(self, inputs):  # pylint: disable=arguments-differ
        # inputs may have extra dimensions (batch_size, d1, ..., dn, sequence_length),
        # but embedding expects (batch_size, sequence_length), so pass inputs to
        # util.combine_initial_dims (which is a no-op if there are no extra dimensions).
        # Remember the original size.
        original_size = inputs.size()
        inputs = util.combine_initial_dims(inputs)

        if self.training and self.word_dropout != 0:
            size = (self.weight.size(0), 1)
            mask = dropout_mask(self.weight, size, self.word_dropout)
            masked_weight = self.weight * mask
        else:
            masked_weight = self.weight

        embedded = embedding(inputs, masked_weight,
                             max_norm=self.max_norm,
                             norm_type=self.norm_type,
                             scale_grad_by_freq=self.scale_grad_by_freq,
                             sparse=self.sparse)

        # Now (if necessary) add back in the extra dimensions.
        embedded = util.uncombine_initial_dims(embedded, original_size)

        if self._projection:
            projection = self._projection
            for _ in range(embedded.dim() - 2):
                projection = TimeDistributed(projection)
            embedded = projection(embedded)

        embedded = self.dropout(embedded)

        return embedded

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Embedding':
        # pylint: disable=arguments-differ
        num_embeddings = params.pop_int('num_embeddings', None)
        vocab_namespace = params.pop("vocab_namespace", "tokens")
        if num_embeddings is None:
            num_embeddings = vocab.get_vocab_size(vocab_namespace)
        embedding_dim = params.pop_int('embedding_dim')
        pretrained_file = params.pop("pretrained_file", None)
        projection_dim = params.pop_int("projection_dim", None)
        trainable = params.pop_bool("trainable", True)
        padding_index = params.pop_int('padding_index', None)
        max_norm = params.pop_float('max_norm', None)
        norm_type = params.pop_float('norm_type', 2.)
        scale_grad_by_freq = params.pop_bool('scale_grad_by_freq', False)
        sparse = params.pop_bool('sparse', False)
        word_dropout = params.pop_float('word_dropout', 0.)
        params.assert_empty(cls.__name__)

        if pretrained_file:
            # If we're loading a saved model, we don't want to actually read a pre-trained
            # embedding file - the embeddings will just be in our saved weights, and we might not
            # have the original embedding file anymore, anyway.
            weight = _read_pretrained_embeddings_file(pretrained_file,
                                                      embedding_dim,
                                                      vocab,
                                                      vocab_namespace)
        else:
            weight = None

        return cls(num_embeddings=num_embeddings,
                   embedding_dim=embedding_dim,
                   projection_dim=projection_dim,
                   weight=weight,
                   padding_index=padding_index,
                   trainable=trainable,
                   max_norm=max_norm,
                   norm_type=norm_type,
                   scale_grad_by_freq=scale_grad_by_freq,
                   sparse=sparse,
                   word_dropout=word_dropout)
