local config_base = import 'config_base.libsonnet';

config_base + {"model": import "../model_configs/show_tell_rnn.libsonnet"}