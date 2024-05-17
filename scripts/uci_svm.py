import subprocess
from typing import Literal

# Set IS_DRAFT to True to print the command without running it
IS_DRAFT = False

# ------------------- Experiment -------------------
tags = "('debug',)"


def run_command(cmd, is_draft=IS_DRAFT):
    print(cmd) if is_draft else subprocess.run(cmd, shell=True)


cmd_template = f"""python main.py \
        --model_config=configs/model.py:"name={{model_class}}" \
        --data_config=configs/data.py:"dataset_name={{dataset_name}}" \
        --optim_config=configs/optim.py:"cooper_optimizer=AlternatingDualPrimalOptimizer {{primal_optim_cmd}} {{dual_optim_cmd}}" \
        --task_config=configs/task.py:"task=max_margin multiplier_kwargs.restart_on_feasible={{restart_on_feasible}}" \
        --metrics_config=configs/metrics.py:"metrics=classification" \
        --resources_config=configs/resources.py:"{{resources_cmd}}" \
        --config.train.total_epochs={{epochs}} \
        --config.logging.wandb_mode={{wandb_mode}} --config.logging.wandb_tags="{{tags}}"
        """


def generate_resources_cmd(
    cluster: Literal["debug", "local", "unkillable", "main", "long", "long_cpu", "long_with_RTX"],
    num_gpus: int = 1,
    timeout_min: int = 60,
    use_ddp: bool = False,
):
    # When no GPU is available, `utils.distributed.init_distributed` ensures that
    # training is run on CPU. You should set `num_gpus=1` in that case anyway.
    if num_gpus > 1:
        assert use_ddp, "use_ddp must be True when num_gpus > 1"
    return f"cluster={cluster} tasks_per_node={num_gpus} timeout_min={timeout_min} use_ddp={use_ddp}"


# ------------------- Resources -------------------
debug_resources = "cluster=local tasks_per_node=1 gpus_per_task=1  timeout_min=90"
wandb_mode = "disabled"
resources_template = f"cluster={{cluster}} nodes=1 tasks_per_node={{num_gpus}} gpus_per_task=1 timeout_min={{timeout_min}} use_ddp={{use_ddp}}"

# # Debug
resources_cmd = debug_resources

# Iris - SVM
model_class = "LinearModel"
dataset_name = "iris"
epochs = 5000
primal_optimizer = "sgd"
primal_lr = 1e-3
dual_lr = 1e-2

primal_optim_cmd_template = f"""primal_optimizer={{primal_optim}} primal_optimizer.kwargs.lr={{primal_lr}}"""
if primal_optimizer == "sgd":
    # Momentum and scheduler
    primal_optim_cmd_template += f""" primal_optimizer.kwargs.momentum=0.9"""  # \
    # primal_optimizer.scheduler_name=StepLR primal_optimizer.scheduler_kwargs.step_size=30 \
    # primal_optimizer.scheduler_kwargs.gamma=0.1"""
primal_optim_cmd = primal_optim_cmd_template.format(primal_optim=primal_optimizer, primal_lr=primal_lr)

#######################################################################################
################################### Experiments #######################################
#######################################################################################

shared_kwargs = dict(
    model_class=model_class,
    dataset_name=dataset_name,
    epochs=epochs,
    primal_optim_cmd=primal_optim_cmd,
    dual_lr=dual_lr,
    wandb_mode=wandb_mode,
    resources_cmd=resources_cmd,
    tags=tags,
)

# Templates for the dual optimizer
kp_template = f"dual_optimizer=pi dual_optimizer.kwargs.lr={{dual_lr}} dual_optimizer.kwargs.Kp={{Kp}}"
sgd_template = f"dual_optimizer=sgd dual_optimizer.kwargs.lr={{dual_lr}} dual_optimizer.kwargs.momentum={{momentum}}"
adam_template = f"dual_optimizer=sparse_adam dual_optimizer.kwargs.lr={{dual_lr}} dual_optimizer.kwargs.betas={{betas}}"

# ------------------- Adam -------------------
if False:
    betas = "(0.9,0.9)"
    dual_optim_cmd = adam_template.format(dual_lr=dual_lr, betas=betas)
    cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, restart_on_feasible=False, **shared_kwargs)
    run_command(cmd)

# ------------------- PI -------------------
Kp_list = []
# Kp_list = [0, 1, 10, 100, 1000, 2000, 5000, 10000]

for Kp in Kp_list:
    restart_on_feasible = False

    dual_optim_cmd = kp_template.format(dual_lr=dual_lr, Kp=Kp)
    cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, restart_on_feasible=restart_on_feasible, **shared_kwargs)
    run_command(cmd)


# ------------------- SGD+M -------------------
momentum_list = []
# momentum_list = [-0.5, -0.1, 0.5, 0.7, 0.9, 0.95]

for momentum in momentum_list:
    restart_on_feasible = False

    dual_optim_cmd = sgd_template.format(dual_lr=dual_lr, momentum=momentum)
    cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, restart_on_feasible=restart_on_feasible, **shared_kwargs)
    run_command(cmd)

# ------------------- With Dual Restarts -------------------
# Vanilla SGD with dual restarts

# if True:
if False:
    restart_on_feasible = True
    dual_optim_cmd = sgd_template.format(dual_lr=dual_lr, momentum=0.0)

    cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, restart_on_feasible=restart_on_feasible, **shared_kwargs)
    run_command(cmd)
