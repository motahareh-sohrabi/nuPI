import re

import cooper
import ml_collections as mlc
import torch

import shared
from src.optim import SGD, nuPI
from src.optim.nuPI_optimizer import InitType as nuPIInitType

MLC_PH = mlc.config_dict.config_dict.placeholder


def _basic_config():
    _config = mlc.ConfigDict()
    _config.name = MLC_PH(str)
    _config.optimizer_class = MLC_PH(type)
    _config.kwargs = mlc.ConfigDict()

    _config.scheduler_name = MLC_PH(str)
    _config.scheduler_kwargs = mlc.ConfigDict()

    return _config


def sgd_config():
    _config = _basic_config()
    _config.name = "SGD"
    _config.optimizer_class = SGD

    _config.kwargs.lr = 1e-2
    _config.kwargs.momentum = 0.0
    _config.kwargs.nesterov = False
    _config.kwargs.maximize = False

    return _config


def nupi_config():
    _config = _basic_config()
    _config.name = "nuPI"
    _config.optimizer_class = nuPI

    _config.kwargs.lr = 1e-2
    _config.kwargs.Kp = 0.0
    _config.kwargs.Ki = 1.0
    _config.kwargs.ema_nu = 0.0
    _config.kwargs.init_type = nuPIInitType.SGD
    _config.kwargs.maximize = True
    _config.kwargs.use_bias_correction = True

    return _config


def adam_config():
    _config = mlc.ConfigDict()
    _config.name = "Adam"
    _config.optimizer_class = torch.optim.Adam

    _config.kwargs = mlc.ConfigDict()
    _config.kwargs.lr = 1e-4
    _config.kwargs.betas = (0.9, 0.999)

    return _config


def sparse_adam_config():
    _config = mlc.ConfigDict()
    _config.name = "SparseAdam"
    _config.optimizer_class = torch.optim.SparseAdam

    _config.kwargs = mlc.ConfigDict()
    _config.kwargs.lr = 1e-4
    _config.kwargs.betas = (0.9, 0.999)

    return _config


OPTIMIZER_CONFIGS = {"sgd": sgd_config, "nupi": nupi_config, "adam": adam_config, "sparse_adam": sparse_adam_config}


def get_config(config_string=None):
    # We expect three mandatory keys:
    #   "cooper_optimizer", "primal_optimizer", and "dual_optimizer"
    # Other keys are optional and depend on the optimizer classes
    # For example, a valid config string is:
    #   "cooper_optimizer=AlternatingDualPrimalOptimizer primal_optimizer=sgd dual_optimizer=sgd"

    # Extract the key-value pairs from the config string which has the format
    # "key1=value1 key2=value2 ..."
    matches = re.findall(shared.REGEX_PATTERN, config_string)

    # Create a dictionary to store the extracted values
    variables = {key: value for key, value in matches}

    # This can be SimultaneousOptimizer or AlternatingDualPrimalOptimizer, etc
    cooper_optimizer = variables.pop("cooper_optimizer")
    cooper_optimizer_class = getattr(cooper.optim, cooper_optimizer)

    primal_optimizer = variables.pop("primal_optimizer")
    primal_config = OPTIMIZER_CONFIGS[primal_optimizer]()
    primal_config_dict = {"params": primal_config}

    dual_optimizer = variables.pop("dual_optimizer")
    dual_config = OPTIMIZER_CONFIGS[dual_optimizer]()
    dual_config.kwargs.maximize = True

    if "gates_optimizer" in variables:
        gates_optimizer = variables.pop("gates_optimizer")
        gates_config = OPTIMIZER_CONFIGS[gates_optimizer]()
        primal_config_dict["gates"] = gates_config

    # Populate options for the primal and dual optimizers
    for key, value in variables.items():
        if key.startswith("primal_optimizer"):
            trimmed_key = key[len("primal_optimizer") + 1 :]
            try:
                value = eval(value)
            except:
                pass
            shared.drill_to_key_and_set(primal_config, key=trimmed_key, value=value)
        elif key.startswith("dual_optimizer"):
            trimmed_key = key[len("dual_optimizer") + 1 :]
            shared.drill_to_key_and_set(dual_config, key=trimmed_key, value=eval(value))
        elif key.startswith("gates_optimizer"):
            trimmed_key = key[len("gates_optimizer") + 1 :]
            shared.drill_to_key_and_set(gates_config, key=trimmed_key, value=eval(value))
        else:
            raise ValueError(f"Unknown optimizer {key} in config string.")

    config = {
        "optim.cooper_optimizer": cooper_optimizer_class,
        "optim.primal_optimizer": primal_config_dict,
        "optim.dual_optimizer": dual_config,
    }

    return config
