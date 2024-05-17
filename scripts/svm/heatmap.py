import itertools

import numpy as np

import utils


def main(
    tags, wandb_mode, resources_cmd, epochs, primal_optim_cmd, all_nupi, all_momentum, all_adam, all_dual_restarts
):
    all_commands = []

    model_class = "LinearModel"
    dataset_name = "iris"

    cmd_template = f"""python main.py \
            --model_config=configs/model.py:"name={{model_class}}" \
            --data_config=configs/data.py:"dataset_name={{dataset_name}}" \
            --optim_config=configs/optim.py:"cooper_optimizer=AlternatingDualPrimalOptimizer {{primal_optim_cmd}} {{dual_optim_cmd}}" \
            --task_config=configs/task.py:"task=max_margin multiplier_kwargs.restart_on_feasible={{restart_on_feasible}}" \
            --metrics_config=configs/metrics.py:"metrics=classification" \
            --resources_config=configs/resources.py:"{{resources_cmd}}" \
            --config.train.total_epochs={{epochs}} \
            --config.logging.wandb_mode={{wandb_mode}} --config.logging.wandb_tags="{{tags}}" \
            --config.logging.log_level="ERROR" --config.allow_silent_failure=True
            """

    ################################### Experiments #######################################
    shared_kwargs = dict(
        model_class=model_class,
        dataset_name=dataset_name,
        epochs=epochs,
        primal_optim_cmd=primal_optim_cmd,
        wandb_mode=wandb_mode,
        resources_cmd=resources_cmd,
        tags=tags,
    )

    # ------------------- nuPI -------------------
    def generate_nupi_cmd(lr, Ki, Kp, ema_nu):
        return f"dual_optimizer=nupi dual_optimizer.kwargs.lr={lr} dual_optimizer.kwargs.Ki={Ki} dual_optimizer.kwargs.Kp={Kp} dual_optimizer.kwargs.ema_nu={ema_nu}"

    for nupi_config in all_nupi:
        dual_optim_cmd = generate_nupi_cmd(**nupi_config)
        cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, restart_on_feasible=False, **shared_kwargs)
        all_commands.append(cmd)

    # ------------------- SGD -------------------
    def generate_sgd_cmd(lr, momentum, nesterov):
        return f"dual_optimizer=sgd dual_optimizer.kwargs.lr={lr} dual_optimizer.kwargs.momentum={momentum} dual_optimizer.kwargs.nesterov={nesterov}"

    # Momentum
    for momentum_config in all_momentum:
        dual_optim_cmd = generate_sgd_cmd(**momentum_config)
        cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, restart_on_feasible=False, **shared_kwargs)
        all_commands.append(cmd)

    # SGD (no momentum) with dual restarts
    for dual_restart_config in all_dual_restarts:
        dual_optim_cmd = generate_sgd_cmd(momentum=0.0, nesterov=False, **dual_restart_config)
        cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, restart_on_feasible=True, **shared_kwargs)
        all_commands.append(cmd)

    # ------------------- Adam -------------------
    def generate_adam_cmd(lr):
        return f"dual_optimizer=sparse_adam dual_optimizer.kwargs.lr={lr}"

    for adam_config in all_adam:
        dual_optim_cmd = generate_adam_cmd(**adam_config)
        cmd = cmd_template.format(dual_optim_cmd=dual_optim_cmd, restart_on_feasible=False, **shared_kwargs)
        all_commands.append(cmd)

    return all_commands


if __name__ == "__main__":
    args = utils.parse_args()

    tags = "('final_svm_heatmap',)"
    wandb_mode = "online"
    resources_cmd = utils.generate_resources_cmd(cluster="debug", num_gpus=1, timeout_min=30, use_ddp=False)

    epochs = 5_000

    primal_optim_cmd = (
        f"""primal_optimizer="sgd" primal_optimizer.kwargs.lr=1e-3 primal_optimizer.kwargs.momentum=0.9"""
    )

    # 10 even log-spaced values between 1e-5 and 1e1
    LOG_DUAL_LRs = np.linspace(-5, 1.5, 20, endpoint=True)
    DUAL_LRs = 10**LOG_DUAL_LRs

    NU_list = [0.0, -0.5, 0.9]
    KP_list = [-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # 18 values

    nupi_prod = itertools.product(DUAL_LRs, KP_list, NU_list)
    all_nupi = [dict(lr=1.0, Ki=Ki, Kp=Kp, ema_nu=ema_nu) for Ki, Kp, ema_nu in nupi_prod]

    momentum_types = ["polyak", "nesterov"]
    momentum_values = []
    momentum_prod = itertools.product(DUAL_LRs, momentum_values, momentum_types)
    all_momentum = [dict(lr=lr, momentum=m, nesterov=(m_type == "nesterov")) for lr, m, m_type in momentum_prod]

    all_adam = []

    all_dual_restarts = []

    ALL_COMMANDS = main(
        tags,
        wandb_mode,
        resources_cmd,
        epochs,
        primal_optim_cmd,
        all_nupi,
        all_momentum,
        all_adam,
        all_dual_restarts,
    )

    for cmd_ix, cmd in enumerate(ALL_COMMANDS):
        print(f"CMD[{cmd_ix}/{len(ALL_COMMANDS)}]: {cmd}")
        print("-----------------------------------------")

    print(f"Total {len(ALL_COMMANDS)} commands")

    if not args.draft:
        utils.call_distributed(ALL_COMMANDS, max_tasks=5)
