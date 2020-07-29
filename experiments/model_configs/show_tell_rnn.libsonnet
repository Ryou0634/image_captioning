local config_base = import '../config_base.libsonnet';
local model_size = 512;

{
    "type": "image_caption_generator",
    "target_embedder": {
        "type": "basic",
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": model_size,
                "trainable": true,
                "vocab_namespace": "tgt_tokens",
            }
        },
    },
    "tgt_vocab_namespace": "tgt_tokens",
    "image_feature_processor": {"type": "fixed_length", "input_size": 25088, "output_size": model_size},
    "decoder": {"type": "rnn_decoder", "input_size": model_size, "hidden_size": model_size, "num_layers": 2},
    "initializer": {"regexes": [[".*", {"type": "uniform", "a": -0.1, "b": 0.1}]], "prevent_regexes": [".*bias.*"]},
    "label_smoothing": 0.0,
    "tie_target_weights": true,
    "beam_size": 5,
    "max_decoding_length": 50

}
