{
    "train_data_path": "data/flickr30k_splits/train.txt",
    "validation_data_path": "data/flickr30k_splits/val.txt",
    "test_data_path": "data/flickr30k_splits/test.txt",
    "dataset_reader": {
        "type": "flickr",
        "caption_file_path": "/data/11/corpora/flickr30k/flickr30k/results_20130124.token",
        "image_feature_path": "data/flickr30k_image_features/vgg16_layer31.npy",
        "image_feature_filename_path": "data/flickr30k_image_features/image_filenames_vgg16_layer31.txt",
    },
    "vocabulary": {"type": "from_instances", "tokens_to_add": {["tgt_tokens"]: ["@start@", "@end@",],},},
    "data_loader": {
        "batch_sampler": {"type": "bucket", "batch_size": 32, "padding_noise": 0.1, "sorting_keys": ["target_tokens"]}
    },
    "trainer": {
        "num_epochs": 50,
        "patience": 3,
        "cuda_device": -1,
        "grad_norm": 5.0,
        "validation_metric": "+bleu",
        "optimizer": {"type": "adam", "lr": 0.001},
        "learning_rate_scheduler": {
            "type": "reduce_on_plateau",
            "mode": "min",
            "factor": 0.5,
            "patience": 1,
            "verbose": true,
        },
        "epoch_callbacks": [
            {
                "type": "log_tokens",
                "input_name_spaces": {"target_tokens": "tgt_tokens"},
                "output_name_spaces": {"predicted_tokens": "tgt_tokens"},
            }
        ],
    },
}
