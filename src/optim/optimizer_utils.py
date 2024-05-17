from types import SimpleNamespace
from typing import Optional

import cooper
import torch

import shared

from .sgd import SGD

logger = shared.fetch_main_logger()


CooperOptimizer = cooper.optim.ConstrainedOptimizer | cooper.optim.UnconstrainedOptimizer


def build_cooper_optimizer_and_schedulers(model, cmp, config) -> tuple[CooperOptimizer, SimpleNamespace]:
    num_params_in_primal_optimizers = 0
    optimizer_config = config.optim.primal_optimizer

    parameters = model.parameter_groups()
    primal_optimizers, primal_schedulers = [], []
    num_primal_params, num_params_in_primal_optimizers = 0, 0

    for group_name, param_group in parameters.items():
        # For Iterator parameter groups, the parameter count below would exhaust the
        # iterator, and the primal optimizer would find no parameters to optimize.
        # Converting to a list ensures that we can loop over the parameters multiple
        # times.
        param_group = list(param_group)

        if group_name not in optimizer_config.keys() or len(param_group) == 0:
            logger.info(f"Using no primal optimizer for group {group_name}")
            continue

        num_params_in_group = sum([param.numel() for param in param_group])
        logger.info(f"  - {group_name}: {num_params_in_group} parameters")
        num_primal_params += num_params_in_group

        primal_optimizer, primal_scheduler, num_params_in_optimizer = build_optimizer_and_scheduler(
            param_group, optimizer_config[group_name]
        )

        primal_optimizers.append(primal_optimizer)
        if primal_scheduler is not None:
            logger.info(f"Created scheduler for group {group_name}")
            primal_schedulers.append(primal_scheduler)
        num_params_in_primal_optimizers += num_params_in_optimizer

    logger.info(
        f"Created optimizers account for {num_params_in_primal_optimizers}/{num_primal_params} primal parameters"
    )

    if cmp.has_dual_variables:
        num_params_in_dual_optimizers = 0
        optimizer_config = config.optim.dual_optimizer
        multiplier = cmp.constraint_group.multiplier

        extra_kwargs = {}
        if isinstance(multiplier, cooper.multipliers.IndexedMultiplier):
            # Pytorch 2.0 does not support `foreach` computation for SGD on parameters
            # with sparse gradients. Hard-coding `foreach=False` as SGD would produce
            # a an error otherwise: "Could not run 'aten::_foreach_neg' with arguments
            # from the 'SparseCUDA' backend".
            # See: https://github.com/pytorch/pytorch/blob/18f203a5678f1d29c4f3f8eecfee95f2206ad5ae/torch/optim/sgd.py#L326
            #
            # As this may be different for other optimizers, we include a NotImplementedError
            if optimizer_config.optimizer_class in [torch.optim.SGD, SGD]:
                extra_kwargs["foreach"] = False

        parameters = cmp.dual_parameter_groups()["multipliers"]
        dual_optimizer, dual_scheduler, num_params_in_optimizer = build_optimizer_and_scheduler(
            parameters, optimizer_config, extra_kwargs=extra_kwargs
        )
        num_params_in_dual_optimizers += num_params_in_optimizer
        logger.info(f"Instantiated the dual optimizer")

        num_dual_params = [sum([_.numel() for _ in group]) for group in cmp.dual_parameter_groups().values()]
        num_dual_params = sum(num_dual_params)
        logger.info(f"Created optimizers account for {num_params_in_dual_optimizers}/{num_dual_params} dual parameters")

        cooper_optimizer = config.optim.cooper_optimizer(
            primal_optimizers, dual_optimizer, multipliers=cmp.constraint_group.multiplier
        )
    else:
        if primal_optimizer is None:
            raise ValueError("UnconstrainedOptimizer expects a primal optimizer but none was provided")
        cooper_optimizer = cooper.optim.UnconstrainedOptimizer(primal_optimizer)

    logger.info(f"Created Cooper optimizer {cooper_optimizer.__class__.__name__}")

    schedulers = SimpleNamespace(primal=primal_schedulers, dual=[dual_scheduler])

    return cooper_optimizer, schedulers


def build_optimizer_and_scheduler(parameters, optimizer_config, extra_kwargs: Optional[dict] = {}):
    optimizer_class = optimizer_config.optimizer_class

    group_names = ["params"]
    optimizer = optimizer_class(parameters, **optimizer_config.kwargs, **extra_kwargs)

    total_params_in_optimizer = 0
    logger.info(f"Created {optimizer_config.name} optimizer for the following groups:")
    for group_name, param_group in zip(group_names, optimizer.param_groups):
        num_params_in_group = sum([param.numel() for param in param_group["params"]])
        logger.info(f"  - {group_name}: {num_params_in_group} parameters")
        total_params_in_optimizer += num_params_in_group

    if "scheduler_name" in optimizer_config and optimizer_config.scheduler_name is not None:
        scheduler_class = torch.optim.lr_scheduler.__dict__[optimizer_config.scheduler_name]
        scheduler = scheduler_class(optimizer, **optimizer_config.scheduler_kwargs)
        logger.info(f"Created {optimizer_config.scheduler_name} scheduler")
    else:
        scheduler = None

    return optimizer, scheduler, total_params_in_optimizer
