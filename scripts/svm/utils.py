import argparse
import subprocess
from multiprocessing.dummy import Pool
from typing import Literal


def call_script(args):
    subprocess.check_call(args, shell=True)


def call_distributed(all_commands, max_tasks=2):
    pool = Pool(max_tasks)
    pool.map(call_script, all_commands)
    pool.close()
    pool.join()


def generate_resources_cmd(
    cluster: Literal["debug", "local", "unkillable", "main", "long", "long_cpu", "long_with_RTX"],
    num_gpus: int = 1,
    cpus_per_task: int = None,
    timeout_min: int = 60,
    use_ddp: bool = False,
):
    # When no GPU is available, `utils.distributed.init_distributed` ensures that
    # training is run on CPU. You should set `num_gpus=1` in that case anyway.
    if num_gpus > 1:
        assert use_ddp, "use_ddp must be True when num_gpus > 1"
    return f"cluster={cluster} tasks_per_node={num_gpus} cpus_per_task={cpus_per_task} timeout_min={timeout_min} use_ddp={use_ddp}"


def parse_args():
    # Add flag for whether to run in draft mode
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", action="store_true")
    args = parser.parse_args()
    return args
