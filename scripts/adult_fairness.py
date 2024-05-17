import itertools
import subprocess
from typing import Literal

# Set IS_DRAFT to True to print the command without running it
IS_DRAFT = False

# ------------------- Experiment -------------------
tags = "('adult', 'sex-race', 'equality', 'epoch-level')"


def run_command(cmd, is_draft=IS_DRAFT):
    print(cmd) if is_draft else subprocess.run(cmd, shell=True)


cmd_template = f"""python main.py \
        --model_config=configs/model.py:"{{model_cmd}}" \
        --data_config=configs/data.py:"{{dataset_cmd}}" \
        --optim_config=configs/optim.py:"cooper_optimizer=AlternatingDualPrimalOptimizer {{primal_optim_cmd}} {{dual_optim_cmd}}" \
        --task_config=configs/task.py:"task=fairness" \
        --metrics_config=configs/metrics.py:"metrics=group_classification" \
        --resources_config=configs/resources.py:"{{resources_cmd}}" \
        --config.train.total_epochs={{epochs}} \
        --config.train.seed={{seed}} \
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
debug_resources = "cluster=long tasks_per_node=1 gpus_per_task=1  timeout_min=30"
wandb_mode = "online"
resources_template = f"cluster={{cluster}} nodes=1 tasks_per_node={{num_gpus}} gpus_per_task=1 timeout_min={{timeout_min}} use_ddp={{use_ddp}}"

# # Debug
resources_cmd = debug_resources

# Adult - MLP
model_cmd = """name=MLP init_kwargs.hidden_sizes='(100, 100)'"""
dataset_cmd = f"""dataset_name=adult dataloader_kwargs.split_kwargs.train.batch_size_per_gpu=512 dataloader_kwargs.split_kwargs.val.batch_size_per_gpu=512"""
epochs = 700

#######################################################################################
################################### Experiments #######################################
#######################################################################################
dual_lr = 1e-2

shared_kwargs = dict(
    model_cmd=model_cmd,
    dataset_cmd=dataset_cmd,
    epochs=epochs,
    dual_lr=dual_lr,
    wandb_mode=wandb_mode,
    resources_cmd=resources_cmd,
    tags=tags,
)

# Templates for the dual optimizer
def generate_primal_optimizer_cmd(primal_optim, lr):
    return f"primal_optimizer={primal_optim} primal_optimizer.kwargs.lr={lr}"


def generate_sgd_cmd(lr, momentum, nesterov):
    return f"dual_optimizer=sgd dual_optimizer.kwargs.lr={lr} dual_optimizer.kwargs.momentum={momentum} dual_optimizer.kwargs.nesterov={nesterov}"


def generate_nupi_cmd(lr, Ki, Kp, ema_nu):
    return f"dual_optimizer=nupi dual_optimizer.kwargs.lr={lr} dual_optimizer.kwargs.Ki={Ki} dual_optimizer.kwargs.Kp={Kp} dual_optimizer.kwargs.ema_nu={ema_nu}"


if False:
    # --------------------------------------------------------------------------------------
    #                    Unconstrained Baseline: Primal Adam
    # --------------------------------------------------------------------------------------
    # Choose the best lr for unconstrained primal
    # PRIMAL_LR_LIST = [1, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    PRIMAL_LR_LIST = [1e-2]

    for primal_lr in PRIMAL_LR_LIST:
        primal_optim_cmd = generate_primal_optimizer_cmd(primal_optim="adam", lr=primal_lr)
        # We are running a constrained problem with a 0-LR and 0 multiplier init, so the constraints are not enforced
        # Using the constrained pipeline gives us the same metrics as in a "true" constrained task
        dual_optim_cmd = generate_sgd_cmd(lr=0.0, momentum=0.0, nesterov=False)
        cmd = cmd_template.format(
            primal_optim_cmd=primal_optim_cmd, dual_optim_cmd=dual_optim_cmd, seed=0, **shared_kwargs
        )
        run_command(cmd)


# --------------------------------------------------------------------------------------
#                    Constrained Experiments
# --------------------------------------------------------------------------------------

# We stick to the best Adam config found in the unconstrained setting
PRIMAL_LR = 1e-2
PRIMAL_OPTIM_CMD = generate_primal_optimizer_cmd(primal_optim="adam", lr=PRIMAL_LR)
# --------------------------------------------------------------------------------------
for seed in [0]:

    SHARED_CONSTRAINED_KWARGS = dict(primal_optim_cmd=PRIMAL_OPTIM_CMD, seed=seed, **shared_kwargs)

    # Choice of dual step-sizes
    DUAL_LRs = [3e-2]

    # --------------------------- GA experiments ---------------------------
    # We use nuPI as the base class for running GA experiments since we can realize GA
    # by setting LR=1, Ki=LR and Kp=0
    GA_prod = itertools.product(DUAL_LRs, [0.0], [0.0])

    ALL_GA = [dict(lr=1.0, Ki=Ki, Kp=Kp, ema_nu=ema_nu) for Ki, Kp, ema_nu in GA_prod]

    for nupi_config in ALL_GA:
        dual_optim_cmd = generate_nupi_cmd(**nupi_config)
        cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, **SHARED_CONSTRAINED_KWARGS)
        run_command(cmd)

    # --------------------------- nuPI experiments ---------------------------
    KP_LISTS = {
        0.99: [5, 10, 20],
        0.9: [1, 5, 10],
        0.5: [0.5, 1, 5],
    }

    for nu_value, KP_list in KP_LISTS.items():
        NU_list = [nu_value]

        NUPI_prod = itertools.product(DUAL_LRs, NU_list, KP_list)
        ALL_NUPI = [dict(lr=1.0, Ki=Ki, Kp=Kp, ema_nu=ema_nu) for Ki, ema_nu, Kp in NUPI_prod]

        for nupi_config in ALL_NUPI:
            dual_optim_cmd = generate_nupi_cmd(**nupi_config)
            cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, **SHARED_CONSTRAINED_KWARGS)
            run_command(cmd)

    # TODO: clean up momentum and Adam configs below!
    # # ------------------- SGD+M -------------------

    # # momentum_types = ["polyak", "nesterov"]
    # # momentum_values = [-0.3, 0.3, -0.5, 0.5, -0.7, 0.7, -0.9, 0.9]

    # # momentum_prod = itertools.product(DUAL_LRs, momentum_values, momentum_types)
    # # all_momentum = [dict(lr=lr, momentum=m, nesterov=(m_type == "nesterov")) for lr, m, m_type in momentum_prod]

    # # for momentum_config in all_momentum:
    # #     dual_optim_cmd = generate_sgd_cmd(**momentum_config)
    # #     cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, **SHARED_CONSTRAINED_KWARGS)
    # #     run_command(cmd)

    # # ------------------- Adam -------------------
    # # all_adam = [dict(lr=lr) for lr in DUAL_LRs]
    # # def generate_adam_cmd(lr):
    # #     return f"dual_optimizer=sparse_adam dual_optimizer.kwargs.lr={lr}"

    # # for adam_config in all_adam:
    # #     dual_optim_cmd = generate_adam_cmd(**adam_config)
    # #     cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, **SHARED_CONSTRAINED_KWARGS)
    # #     run_command(cmd)
