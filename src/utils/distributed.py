import os
import socket
from types import SimpleNamespace
from typing import Any, Union

import torch
import torch.distributed as dist

import shared

logger = shared.fetch_main_logger()


def init_distributed(resources_config):
    """
    Initialize torch.distributed if using multiple GPUs.
    Extracts relevant info from SLURM environment variables.
    Sets device based on local_rank.
    Defines {multi_gpu, rank, local_rank, world_size, device}
    """

    if (resources_config.nodes * resources_config.tasks_per_node * resources_config.gpus_per_task) > 1:
        logger.info("Initializing multi-GPU environment")

        multi_gpu = True

        slurm_local_rank = -1
        if not dist.is_initialized():
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

            # These lines were in job.sh
            # export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
            # MASTER_PORT=$((10000 + 10#"$(echo -n "$SLURM_JOBID" | tail -c 4)"))

            slurm_rank = int(os.getenv("SLURM_PROCID"))
            slurm_local_rank = int(os.getenv("SLURM_LOCALID"))
            slurm_world_size = int(os.getenv("SLURM_NTASKS"))

            os.environ["RANK"] = str(slurm_rank)
            os.environ["WORLD_SIZE"] = str(slurm_world_size)

            # os.environ["MASTER_ADDR"] = master_addr
            os.environ["MASTER_PORT"] = "40101"
            dist.init_process_group(backend="nccl", init_method="env://", rank=slurm_rank, world_size=slurm_world_size)

        logger.info("torch.distributed is initialized")

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size > 8:
            assert slurm_local_rank >= 0
            local_rank = slurm_local_rank
        else:
            local_rank = rank
    else:
        logger.info("This is a single GPU job")
        multi_gpu = False
        world_size = 1
        rank = 0
        local_rank = 0

    logger.info(f"Rank {rank}")
    logger.info(f"World size {world_size}")
    logger.info(f"Local rank {local_rank}")
    logger.info(f"Running on host {socket.gethostname()}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return SimpleNamespace(multi_gpu=multi_gpu, rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def wait_for_all_processes():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def do_all_gather_object(object):
    if dist.is_available() and dist.is_initialized():
        output_objects = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(output_objects, object)
        return output_objects
    return [object]


def do_broadcast_object(object):
    objects = [object]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(objects)
    return objects[0]


def do_reduce_mean(data: Union[list[dict[str, Any]], dict[str, Any], torch.Tensor, int], dst_rank: int = 0):
    if not dist.is_available() or not dist.is_initialized():
        return data

    if isinstance(data, torch.Tensor):
        cloned_tensor = data.clone()
        dist.reduce(cloned_tensor, dst=dst_rank, op=dist.ReduceOp.SUM)
        return cloned_tensor / dist.get_world_size()
    elif isinstance(data, int):
        # The original data was integers on each worker, so we extract the value from
        # the tensor to avoid returning a tensor object (as opposed to a float).
        return do_reduce_mean(torch.tensor(data, dtype=torch.int, device="cuda"), dst_rank=dst_rank).item()
    elif isinstance(data, dict):
        return {k: do_reduce_mean(v, dst_rank=dst_rank) for k, v in data.items()}
    elif isinstance(data, list):
        return [do_reduce_mean(v, dst_rank=dst_rank) for v in data]
    else:
        raise ValueError(f"Unsupported type: {type(data)}")
