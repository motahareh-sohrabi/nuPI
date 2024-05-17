import ml_collections as mlc

import shared

MLC_PH = mlc.config_dict.config_dict.placeholder

# Excluding V100s and cn-g[001-012] (MIG) nodes by default.
DEFAULT_EXCLUDE = "cn-b[001-005],cn-e[002-003],cn-g[001-012],cn-j001"
RTX_EXCLUDE = "cn-a[001-011],cn-c[001-040]"


def _basic_config():
    resources_config = mlc.ConfigDict()
    resources_config.cluster = MLC_PH(str)  # "slurm", "local" or "debug"
    resources_config.partition = MLC_PH(str)
    resources_config.nodes = 1
    resources_config.timeout_min = MLC_PH(int)
    resources_config.tasks_per_node = MLC_PH(int)
    resources_config.cpus_per_task = MLC_PH(int)
    resources_config.gpus_per_task = 1
    resources_config.mem = MLC_PH(str)
    resources_config.exclude = MLC_PH(str)
    resources_config.use_ddp = False
    return resources_config


def debug_config():
    """This config is used for running experiments on the main Python thread. You may
    want to use this for:
        - debugging your code locally
        - running a single experiment on your local machine
    """
    resources_config = _basic_config()
    resources_config.cluster = "debug"
    resources_config.tasks_per_node = 1  # Multi-GPU debugging is not supported.
    resources_config.use_ddp = False
    return resources_config


def local_config():
    """This config is used for running experiments _locally_, but not on the main Python
    thread. You may want to use this for:
        - running a single experiment in the background on your local machine
        - running a single experiment on a remote node
    """
    resources_config = _basic_config()
    resources_config.cluster = "local"
    resources_config.is_slurm_job = False
    # resources_config.tasks_per_node = ??? # Pick the number of GPUs available
    return resources_config


def unkillable_config():
    resources_config = _basic_config()
    resources_config.cluster = "slurm"
    resources_config.partition = "unkillable"
    resources_config.nodes = 1
    resources_config.tasks_per_node = 1
    resources_config.cpus_per_task = 6
    resources_config.gpus_per_task = 1
    resources_config.mem = "32GB"
    resources_config.exclude = DEFAULT_EXCLUDE + "," + RTX_EXCLUDE
    resources_config.use_ddp = False

    return resources_config


def main_config():
    resources_config = _basic_config()
    resources_config.cluster = "slurm"
    resources_config.partition = "main"
    resources_config.nodes = 1
    # resources_config.tasks_per_node = ??? # Pick 1 or 2 GPUs for `main` partition
    resources_config.cpus_per_task = 6
    resources_config.gpus_per_task = 1
    resources_config.mem = "48GB"
    resources_config.exclude = DEFAULT_EXCLUDE + "," + RTX_EXCLUDE

    return resources_config


def long_config():
    resources_config = _basic_config()
    resources_config.cluster = "slurm"
    resources_config.partition = "long"
    resources_config.nodes = 1
    # resources_config.tasks_per_node = ??? # Pick any number of GPUs for `long` partition
    resources_config.cpus_per_task = 12
    resources_config.gpus_per_task = 1
    resources_config.mem = "48GB"
    resources_config.exclude = DEFAULT_EXCLUDE + "," + RTX_EXCLUDE

    return resources_config


def long_with_RTX_config():
    # Same as long_config but with RTX nodes not excluded

    resources_config = _basic_config()
    resources_config.cluster = "slurm"
    resources_config.partition = "long"
    resources_config.nodes = 1
    # resources_config.tasks_per_node = ??? # Pick any number of GPUs for `long` partition
    resources_config.cpus_per_task = 12
    resources_config.gpus_per_task = 1
    resources_config.mem = "48GB"
    resources_config.exclude = DEFAULT_EXCLUDE

    return resources_config


def long_cpu_config():
    resources_config = _basic_config()
    resources_config.cluster = "slurm"
    resources_config.partition = "long-cpu"
    resources_config.nodes = 1
    # resources_config.tasks_per_node = ??? # Pick any number of GPUs for `long` partition
    resources_config.cpus_per_task = 2
    resources_config.gpus_per_task = 0
    resources_config.mem = "16GB"
    resources_config.exclude = DEFAULT_EXCLUDE

    return resources_config


RESOURCES_CONFIGS = {
    None: _basic_config,
    "debug": debug_config,
    "local": local_config,
    "unkillable": unkillable_config,
    "main": main_config,
    "long": long_config,
    "long_with_RTX": long_with_RTX_config,
    "long-cpu": long_cpu_config,
}


def get_config(config_string=None):
    return shared.default_get_config(
        config_key="resources", pop_key="cluster", preset_configs=RESOURCES_CONFIGS, cli_args=config_string
    )
