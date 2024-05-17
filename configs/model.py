import ml_collections as mlc
import torch

import shared
from src import models

MLC_PH = mlc.config_dict.config_dict.placeholder


def _basic_config():
    _config = mlc.ConfigDict()

    _config.model_class = MLC_PH(type)
    _config.init_seed = MLC_PH(int)

    _config.init_kwargs = mlc.ConfigDict()
    _config.init_kwargs.from_trainer = {}

    return _config


def LinearModel_config():
    _config = _basic_config()

    _config.model_class = models.LinearModel
    _config.init_seed = 246

    _config.init_kwargs.bias = True

    return _config


def MLP_config():
    _config = _basic_config()

    _config.model_class = models.MLP
    _config.init_seed = 246

    _config.init_kwargs.activation_type = torch.nn.ReLU
    _config.init_kwargs.hidden_sizes = [100, 100]

    return _config


def CifarSparseResNet18_config():
    _config = _basic_config()

    _config.model_class = models.CifarSparseResNet18
    _config.init_seed = 246

    _config.init_kwargs.sparsity_type = "structured"
    _config.init_kwargs.masked_conv_ix = ["conv1", "conv2"]

    _config.init_kwargs.norm_layer = models.sparse.layers.MaskedBatchNorm2d
    _config.init_kwargs.is_first_conv_dense = True
    _config.init_kwargs.is_last_fc_dense = True

    _config.init_kwargs.masked_layer_kwargs = mlc.ConfigDict()
    _config.init_kwargs.masked_layer_kwargs.droprate_init = 0.01

    return _config


MODEL_CONFIGS = {
    "LinearModel": LinearModel_config,
    "MLP": MLP_config,
    "CifarSparseResNet18": CifarSparseResNet18_config,
}


def get_config(config_string=None):
    return shared.default_get_config(
        config_key="model", pop_key="name", preset_configs=MODEL_CONFIGS, cli_args=config_string
    )
