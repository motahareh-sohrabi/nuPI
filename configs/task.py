import cooper
import ml_collections as mlc
import torch

MLC_PH = mlc.config_dict.config_dict.placeholder

import shared
import src.trainers as trainers
from src.cmp import FairnessConstrainedClassification, MaxMarginClassification, SparsityConstrainedClassification


def _basic_config():
    _config = mlc.ConfigDict()

    _config.trainer_class = MLC_PH(type)
    _config.dtype = torch.float32
    _config.cmp_class = MLC_PH(type)
    _config.constraint_type = MLC_PH(cooper.ConstraintType)

    _config.cmp_kwargs = mlc.ConfigDict()
    _config.cmp_kwargs.from_trainer = {}

    _config.multiplier_kwargs = mlc.ConfigDict()

    # Used to determine whether the datset needs to be wrapped in an IndexedDataset for
    # extracting the right constraint and multiplier indices
    _config.use_indexed_dataset = MLC_PH(bool)

    return _config


def fairness_constrained_classification_task_config():
    _config = _basic_config()

    _config.trainer_class = trainers.FairnessTrainer
    _config.cmp_class = FairnessConstrainedClassification
    _config.constraint_type = cooper.ConstraintType.INEQUALITY

    _config.cmp_kwargs.from_trainer = {"all_protected_groups": "datasets.train.all_protected_groups"}
    _config.cmp_kwargs.tolerance = 0.01

    _config.use_indexed_dataset = False
    _config.multiplier_kwargs.restart_on_feasible = False

    return _config


def sparsity_constrained_classification_task_config():
    _config = _basic_config()

    _config.trainer_class = trainers.SparsityTrainer
    _config.cmp_class = SparsityConstrainedClassification
    _config.constraint_type = cooper.ConstraintType.INEQUALITY

    _config.cmp_kwargs.weight_decay = 0.0
    _config.cmp_kwargs.target_sparsities = MLC_PH(list)
    _config.cmp_kwargs.from_trainer = {}

    _config.use_indexed_dataset = False
    _config.multiplier_kwargs.restart_on_feasible = False
    _config.multiplier_kwargs.init = 0.0

    return _config


def max_margin_task_config():
    _config = _basic_config()

    _config.trainer_class = trainers.SVMTrainer
    _config.dtype = torch.float64
    _config.cmp_class = MaxMarginClassification

    _config.constraint_type = cooper.ConstraintType.INEQUALITY

    _config.cmp_kwargs.lmbda_tolerance = 1e-6
    _config.cmp_kwargs.ensure_LICQ = True

    _config.cmp_kwargs.from_trainer = {"train_dataset": "datasets.train"}

    _config.use_indexed_dataset = True
    _config.multiplier_kwargs.restart_on_feasible = False

    return _config


TASK_CONFIGS = {
    "max_margin": max_margin_task_config,
    "fairness": fairness_constrained_classification_task_config,
    "sparsity": sparsity_constrained_classification_task_config,
}


def get_config(config_string=None):
    return shared.default_get_config(
        config_key="task", pop_key="task", preset_configs=TASK_CONFIGS, cli_args=config_string
    )
