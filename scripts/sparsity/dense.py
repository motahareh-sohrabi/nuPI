"""
CIFAR 10/100 training recipe taken from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
"""

import subprocess

# NOTE: we set the droprate of the model to 0.999, which means that the model is
# effectively initialized to be fully dense. Moreover, we hardcode the target sparsity
# to 0.0, and the gates and dual learning rates to 0.0 to prevent the gates from
# changing. This is a workaround to implementing a dense model and a dense cmp.

cmd_template = f"""python main.py \
    --model_config=configs/model.py:"{{model_cmd}}" \
    --data_config=configs/data.py:"dataset_name={{dataset_name}}" \
    --optim_config=configs/optim.py:"cooper_optimizer=AlternatingDualPrimalOptimizer \
        {{primal_optim_cmd}} gates_optimizer=sgd gates_optimizer.kwargs.lr=0.0 \
        dual_optimizer=sgd dual_optimizer.kwargs.lr=0.0" \
    --task_config=configs/task.py:"task=sparsity cmp_kwargs.target_sparsities=(0.0,) \
        cmp_kwargs.weight_decay={{weight_decay}}" \
    --metrics_config=configs/metrics.py:"metrics=classification" \
    --resources_config=configs/resources.py:"{{resources_cmd}}" \
    --config.train.total_epochs={{epochs}} \
    --config.logging.wandb_mode={{wandb_mode}} \
    --config.logging.wandb_tags="{{tags}}" \
    """

# ------------------- Task -------------------
tags = f"('dense',)"
sparsity_type = "structured"


# ------------------- Resources -------------------
debug_resources = f"""cluster=debug timeout_min=90"""
resources_template = f"""
    cluster={{cluster}} nodes=1 tasks_per_node={{num_gpus}} gpus_per_task=1 \
        timeout_min={{timeout_min}} use_ddp={{use_ddp}}
    """

# # Local
# debug_resources = """cluster=debug timeout_min=90"""
# wandb_mode = "disabled"
# resources_cmd = debug_resources

# For 1 GPU jobs
cluster = "long_with_RTX"
num_gpus = 1
timeout_min = 180
use_ddp = False
resources_cmd = resources_template.format(cluster=cluster, num_gpus=num_gpus, timeout_min=timeout_min, use_ddp=use_ddp)
wandb_mode = "online"


# ------------------- Dataset/Models -------------------
model_cmd_template = f"""name={{model_class}} init_kwargs.sparsity_type={{sparsity_type}}"""

# CIFAR10
model_class = "CifarSparseResNet18"
dataset_name = "cifar10"
epochs = 200
primal_optimizer = "sgd"
primal_lr = 0.1
weight_decay = 5e-4


mask_conv_ix = ()
model_cmd = model_cmd_template.format(model_class=model_class, sparsity_type=sparsity_type)
if model_class in ("CifarSparseResNet18",):
    model_cmd += (
        f""" init_kwargs.masked_conv_ix={{masked_conv_ix}} init_kwargs.masked_layer_kwargs.droprate_init=0.001"""
    )
else:
    model_cmd += f""" init_kwargs.droprate_init=0.001"""

primal_optim_cmd_template = f"""primal_optimizer={{primal_optim}} primal_optimizer.kwargs.lr={{primal_lr}}"""
if primal_optimizer == "sgd":
    # Momentum and scheduler for runs with SGD
    primal_optim_cmd_template += f""" primal_optimizer.kwargs.momentum=0.9 \
        primal_optimizer.scheduler_name=CosineAnnealingLR primal_optimizer.scheduler_kwargs.T_max={epochs}"""

primal_optim_cmd = primal_optim_cmd_template.format(primal_optim=primal_optimizer, primal_lr=primal_lr)

#######################################################################################
################################### Experiments #######################################
#######################################################################################


cmd = cmd_template.format(
    model_cmd=model_cmd,
    dataset_name=dataset_name,
    epochs=epochs,
    primal_optim_cmd=primal_optim_cmd,
    wandb_mode=wandb_mode,
    resources_cmd=resources_cmd,
    sparsity_type=sparsity_type,
    tags=tags,
    weight_decay=weight_decay,
)
subprocess.run(cmd, shell=True)
