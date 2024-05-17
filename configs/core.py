import ml_collections as mlc
import torch

MLC_PH = mlc.config_dict.config_dict.placeholder


def build_basic_config():
    config = mlc.ConfigDict()

    config.exp_name = MLC_PH(str)
    config.allow_silent_failure = False

    config.train = mlc.ConfigDict()
    config.train.seed = 0
    config.train.use_deterministic_ops = True
    config.train.total_epochs = MLC_PH(int)
    config.train.total_steps = MLC_PH(int)

    # These configs rely on separate config_files. See header of `main.py`
    config.model = mlc.ConfigDict()
    config.data = mlc.ConfigDict()
    config.task = mlc.ConfigDict()
    config.metrics = mlc.ConfigDict()
    config.resources = mlc.ConfigDict()

    config.optim = mlc.ConfigDict()
    config.optim.cooper_optimizer = MLC_PH(type)
    config.optim.primal_optimizer = mlc.ConfigDict()
    config.optim.dual_optimizer = mlc.ConfigDict()

    # Fixed defaults for logging across all experiments
    config.logging = mlc.ConfigDict()
    config.logging.log_level = "INFO"
    config.logging.print_train_stats_period_steps = 100
    config.logging.eval_period_epochs = 1
    config.logging.eval_period_steps = MLC_PH(int)
    config.logging.wandb_mode = "disabled"
    config.logging.wandb_tags = MLC_PH(tuple)
    config.logging.run_name = MLC_PH(str)
    config.logging.results = mlc.ConfigDict()

    config.checkpointing = mlc.ConfigDict()
    config.checkpointing.enabled = True
    config.checkpointing.resume_from_checkpoint = False

    return config


def get_config():
    return build_basic_config()
