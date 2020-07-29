local config_base = import 'config_base.libsonnet';

config_base + {"model": {"type": "sequence_reinforce",
                         "sequence_generator": import "../model_configs/show_attend_rnn.libsonnet",
                         "self_critic": true,
                         "initializer": {
                            "regexes": [
                                [
                                    ".*",
                                    {
                                        "type": "pretrained",
                                        "weights_file_path": "results/flickr/show_attend_rnn_best/default/best.th"
                                    },
                                ]
                            ]
                        }}}