import ml_collections as mlc

import shared

MLC_PH = mlc.config_dict.config_dict.placeholder


def _basic_config():
    metrics_config = mlc.ConfigDict()
    metrics_config.train_metrics = MLC_PH(list)
    metrics_config.val_metrics = MLC_PH(list)
    return metrics_config


def classification_config():
    metrics_config = _basic_config()
    metrics_config.train_metrics = [{"name": "Accuracy", "metric": "Accuracy", "kwargs": {"per_sample": False}}]
    metrics_config.val_metrics = [{"name": "Accuracy", "metric": "Accuracy", "kwargs": {"per_sample": False}}]

    return metrics_config


def group_classification_config():
    metrics_config = _basic_config()
    metrics_config.train_metrics = [
        {"name": "Accuracy", "metric": "Accuracy", "kwargs": {"per_sample": False}},
        {"name": "GroupAccuracy", "metric": "Accuracy", "kwargs": {"per_group": True}},
        {"name": "Probability", "metric": "PositiveProbability", "kwargs": {"per_sample": False}},
        {"name": "GroupProbability", "metric": "PositiveProbability", "kwargs": {"per_group": True}},
    ]
    metrics_config.val_metrics = [
        {"name": "Accuracy", "metric": "Accuracy", "kwargs": {"per_sample": False}},
        {"name": "GroupAccuracy", "metric": "Accuracy", "kwargs": {"per_group": True}},
        {"name": "Probability", "metric": "PositiveProbability", "kwargs": {"per_sample": False}},
        {"name": "GroupProbability", "metric": "PositiveProbability", "kwargs": {"per_group": True}},
    ]

    return metrics_config


METRICS_CONFIGS = {
    "classification": classification_config,
    "group_classification": group_classification_config,
    None: _basic_config,
}


def get_config(config_string=None):
    return shared.default_get_config(
        config_key="metrics", pop_key="metrics", preset_configs=METRICS_CONFIGS, cli_args=config_string
    )
