import cooper
import torch
import torch.nn.functional as F

import shared
import src.models.sparse.layers as layers

from .base import BaseProblem

logger = shared.fetch_main_logger()


class SparsityConstrainedClassification(BaseProblem):

    has_dual_variables = True

    def __init__(self, weight_decay: float, target_sparsities: list[float], multiplier_kwargs: dict = {}):
        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")

        target_sparsities = [target_sparsities] if not isinstance(target_sparsities, list) else target_sparsities
        logger.info(f"Setting up SparsityConstrainedClassification problem with target_sparsities={target_sparsities}")

        self.weight_decay = weight_decay

        self.constraint_type = cooper.ConstraintType.INEQUALITY
        self.num_constraints, self.multiplier = self.create_multiplier(
            target_sparsities, multiplier_kwargs=multiplier_kwargs
        )

        self.constraint_group = cooper.ConstraintGroup(
            constraint_type=self.constraint_type,
            formulation_type=cooper.FormulationType.LAGRANGIAN,
            multiplier=self.multiplier,
        )

        target_sparsities = torch.tensor(target_sparsities, device=self.multiplier.device)
        assert torch.all(target_sparsities >= 0) and torch.all(target_sparsities <= 1)
        self.target_densities = 1 - target_sparsities

    def create_multiplier(self, target_sparsities, multiplier_kwargs):
        num_constraints = len(target_sparsities)
        multiplier = cooper.multipliers.DenseMultiplier(
            constraint_type=self.constraint_type, num_constraints=num_constraints, **multiplier_kwargs
        )
        return num_constraints, multiplier

    def compute_sparsity_stats(self, model, is_test_time: bool = False):
        model_stats = layers.get_model_stats(model, is_test_time=is_test_time)
        model_density = model_stats.compute_density_stats()[1]

        layer_densities = []
        for layer_stats in model_stats.layer_stats:
            layer_density = layer_stats.compute_density_stats()[1]
            if layer_density is not None:
                layer_densities.append(layer_density)

        if len(layer_densities) > 0:
            layer_densities = torch.stack(layer_densities)
        else:
            # Models with dense layers do not produce layer_density measurements. In
            # practice, their density is always 1.0.
            layer_densities = torch.ones_like(self.target_densities)

            # Model density is always 1.0 for models with dense layers
            model_density = torch.tensor([1.0], device=self.target_densities.device)

        squared_l2_norm = model_stats.sq_l2_norm

        return model_density, layer_densities, squared_l2_norm

    def compute_cmp_state(self, model, inputs, targets) -> cooper.CMPState:
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="mean")

        model_density, layer_densities, squared_l2_norm = self.compute_sparsity_stats(model, is_test_time=False)

        if self.weight_decay != 0:
            # model_stats.sq_l2_norm is the (expected) squared l2 norm of the weights
            # and biases
            loss += 0.5 * squared_l2_norm * self.weight_decay

        # Computing the violation of the sparsity constraint
        if self.target_densities.numel() == 1:
            sparsity_violation = model_density - self.target_densities
        else:
            sparsity_violation = layer_densities - self.target_densities

        constraint_state = cooper.ConstraintState(violation=sparsity_violation)

        batch_log_metrics = dict(
            loss=loss,
            model_density=model_density,
            layer_densities=layer_densities,
            squared_l2_norm=squared_l2_norm,
            predictions=logits,
        )

        return cooper.CMPState(
            loss=loss, observed_constraints=[(self.constraint_group, constraint_state)], misc=batch_log_metrics
        )

    def dual_parameter_groups(self):
        return {"multipliers": self.multiplier.parameters()}
