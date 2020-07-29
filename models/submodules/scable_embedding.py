import torch
from allennlp.modules.token_embedders import TokenEmbedder, Embedding
from allennlp.modules.token_embedders.embedding import _read_pretrained_embeddings_file

from allennlp.data.vocabulary import Vocabulary
from allennlp.common import Params


@TokenEmbedder.register("scalable")
class ScalableEmbedding(Embedding):
    """
    Used for transformer.
    Scale output embedding by embed_dim ** 0.5.
    """

    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 vocab_namespace: str,
                 padding_index: int = None,
                 trainable: bool = True,
                 pretrained_file: str = None,
                 scale: bool = False,
                 projection_dim: int = None,
                 weight: torch.FloatTensor = None,
                 max_norm: float = None,
                 norm_type: float = 2.,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False) -> None:
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            weight=weight,
            padding_index=padding_index,
            trainable=trainable,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            vocab_namespace=vocab_namespace,
            pretrained_file=pretrained_file
        )

        self.scale_factor = self.weight.size(1) ** 0.5
        self.scale = scale

    def forward(self, inputs: torch.LongTensor) -> torch.Tensor:
        output = super().forward(inputs)

        if self.scale:
            output *= self.scale_factor

        return output

    # Custom logic requires custom from_params.
    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'Embedding':  # type: ignore
        # pylint: disable=arguments-differ
        num_embeddings = params.pop_int('num_embeddings', None)
        # If num_embeddings is present, set default namespace to None so that extend_vocab
        # call doesn't misinterpret that some namespace was originally used.
        vocab_namespace = params.pop("vocab_namespace", None if num_embeddings else "tokens")
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
        scale = params.pop_bool('scale', False)
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
                   scale=scale,
                   vocab_namespace=vocab_namespace)
