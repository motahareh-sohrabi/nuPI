import subprocess

cmd_template = f"""python main.py \
    --model_config=configs/model.py:"{{model_cmd}}" \
    --data_config=configs/data.py:"dataset_name={{dataset_name}}" \
    --optim_config=configs/optim.py:"cooper_optimizer=AlternatingDualPrimalOptimizer \
        {{primal_optim_cmd}} gates_optimizer=adam gates_optimizer.kwargs.lr={{gates_lr}} \
        {{dual_optim_cmd}}" \
    --task_config=configs/task.py:"task=sparsity cmp_kwargs.target_sparsities={{ts}} \
        multiplier_kwargs.restart_on_feasible={{restart_on_feasible}} \
        cmp_kwargs.weight_decay={{weight_decay}}" \
    --metrics_config=configs/metrics.py:"metrics=classification" \
    --resources_config=configs/resources.py:"{{resources_cmd}}" \
    --config.train.total_epochs={{epochs}} \
    --config.train.seed={{seed}} \
    --config.logging.wandb_mode={{wandb_mode}} \
    --config.logging.wandb_tags="{{tags}}" \
    """

# ------------------- Task -------------------
tags = f"('icml',)"
seeds = [0, 1, 2, 3, 4]

# sparsity_type = "unstructured"
sparsity_type = "structured"
droprate_init = 0.01

# --------- Model-wise target
target_sparsities = ["(0.3,)", "(0.5,)", "(0.7,)", "(0.9,)"]

# --------- Layer-wise target
# ResNet18 layer-wise target. ResNet18 models have 8 sparsifiable conv layers.
target_sparsities += [
    "(0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3)",
    "(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)",
    "(0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7)",
    "(0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9)",
]

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
model_cmd_template = f"""name={{model_class}} init_kwargs.sparsity_type={{sparsity_type}} init_kwargs.masked_layer_kwargs.droprate_init={{droprate_init}}"""

# CIFAR10
model_class = "CifarSparseResNet18"
# # Uncomment for unstructured sparsity
# masked_conv_ix = "('conv1', 'conv2')"
# Uncomment for structured sparsity
masked_conv_ix = "('conv1',)"
dataset_name = "cifar10"
epochs = 200
primal_optimizer = "sgd"
primal_lr = 0.1
gates_lr = 1e-3
dual_lr = 8e-4
weight_decay = 5e-4

model_cmd = model_cmd_template.format(model_class=model_class, sparsity_type=sparsity_type, droprate_init=droprate_init)

if model_class in ("CifarSparseResNet18",):
    model_cmd += f""" init_kwargs.masked_conv_ix={masked_conv_ix} """

primal_optim_cmd_template = f"""primal_optimizer={{primal_optim}} primal_optimizer.kwargs.lr={{primal_lr}}"""
if primal_optimizer == "sgd":
    # Momentum and scheduler for runs with SGD
    primal_optim_cmd_template += f""" primal_optimizer.kwargs.momentum=0.9 \
        primal_optimizer.scheduler_name=CosineAnnealingLR primal_optimizer.scheduler_kwargs.T_max={epochs}"""

primal_optim_cmd = primal_optim_cmd_template.format(primal_optim=primal_optimizer, primal_lr=primal_lr)


# -------------------------- Kp Choices --------------------------
Kp_per_sparsity = {
    "(0.3,)": [12_000],
    "(0.5,)": [12_000],
    "(0.7,)": [18_000],
    "(0.9,)": [2_000],
    "(0.3,0.3,0.3,0.3,0.3,0.3,0.3,0.3)": [10_000],
    "(0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5)": [10_000],
    "(0.7,0.7,0.7,0.7,0.7,0.7,0.7,0.7)": [15_000],
    "(0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9)": [12_000],
}

#######################################################################################
################################### Experiments #######################################
#######################################################################################

for seed in seeds:
    for target_sparsity in target_sparsities:
        shared_kwargs = dict(
            model_cmd=model_cmd,
            dataset_name=dataset_name,
            epochs=epochs,
            primal_optim_cmd=primal_optim_cmd,
            gates_lr=gates_lr,
            dual_lr=dual_lr,
            wandb_mode=wandb_mode,
            resources_cmd=resources_cmd,
            sparsity_type=sparsity_type,
            ts=target_sparsity,
            weight_decay=weight_decay,
            tags=tags,
            seed=seed,
        )

        # Templates for the dual optimizer
        kp_template = f"dual_optimizer=pi dual_optimizer.kwargs.lr={{dual_lr}} dual_optimizer.kwargs.Kp={{Kp}}"

        # ------------------- PI -------------------
        Kp_list = Kp_per_sparsity[target_sparsity]

        for Kp in Kp_list:
            dual_optim_cmd = kp_template.format(dual_lr=dual_lr, Kp=Kp)
            cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, restart_on_feasible=False, **shared_kwargs)
            subprocess.run(cmd, shell=True)
