import cooper
import numpy as np
import torch
import torch.nn.functional as F

import shared

from .base import BaseProblem

logger = shared.fetch_main_logger()


class FairnessConstrainedClassification(BaseProblem):
    """
    Implements the following fairness-constrained classification problem:
    - min L(w) + 0.5 * weight_decay * ||w||^2
    - s.t. P(\hat{y} = 1 | G = g) = P(\hat{y} = 1) for all groups g

    We work under a bernoulli model, where our model predicts P(\hat{y} = 1 | x), the
    probability of sample x belonging to class 1. Here, L(w) is the cross-entropy loss
    of model w, and G represents a sensitive attribute.

    The constraint asks that the probability of predicting class 1 for a given group g
    is equal to the overall probability of predicting class 1.
    """

    has_dual_variables = True

    def __init__(self, all_protected_groups: list, tolerance: float = 0.0, multiplier_kwargs: dict = {}):

        logger.info(f"Setting up FairnessConstrainedClassification problem")

        assert tolerance >= 0, "Tolerance must be non-negative"
        self.constraint_type = cooper.ConstraintType.EQUALITY

        logger.info(f"Using {self.constraint_type} constraints with tolerance {tolerance}")

        self.all_protected_groups = all_protected_groups
        self.num_groups = len(all_protected_groups)
        self.num_constraints = len(all_protected_groups)

        self.tolerance = tolerance

        self.multiplier = self.create_multiplier(multiplier_kwargs)

        self.constraint_group = cooper.ConstraintGroup(
            constraint_type=self.constraint_type,
            formulation_type=cooper.FormulationType.LAGRANGIAN,
            multiplier=self.multiplier,
        )

    def compute_constraints(self, logits, groups) -> cooper.ConstraintState:
        """
        Compute the fairness constraints on the provided batch of data.
        If a group is not observed in the batch, we do not compute a constraint for it.
        """
        sample_probs = F.sigmoid(logits).squeeze()  # shape: (batch_size,)
        aggregate_prob = torch.mean(sample_probs)

        groups = np.array(groups)
        observed_groups = []
        group_probs = []
        for g_idx, g in enumerate(self.all_protected_groups):
            group_mask = groups == g
            if group_mask.sum() == 0:
                # Group g is not observed in this batch
                continue
            else:
                observed_groups.append(g_idx)
                group_probs.append(torch.mean(sample_probs[group_mask]))

        group_probs = torch.stack(group_probs)
        observed_groups = torch.tensor(observed_groups, dtype=torch.long, device=group_probs.device)

        constraint_measurements = group_probs - aggregate_prob

        return cooper.ConstraintState(
            violation=constraint_measurements, constraint_features=observed_groups, contributes_to_dual_update=False
        )

    def compute_cmp_state(self, model, inputs, targets, groups) -> cooper.CMPState:
        """
        Compute the loss and fairness constraints for the given model on the provided batch.
        NOTE: assuming a *binary* classification problem for now.
        """
        logits = model(inputs)  # shape: (batch_size, 1)
        assert logits.shape[1] == 1, "FairnessConstrainedClassification only supports binary classification for now"

        loss = F.binary_cross_entropy_with_logits(logits.squeeze(), targets.float())
        accuracy = (F.sigmoid(logits).squeeze().round() == targets).float().mean()
        constraint_state = self.compute_constraints(logits, groups)
        misc = {"predictions": logits, "loss": loss, "accuracy": accuracy}

        return cooper.CMPState(loss=loss, observed_constraints=[(self.constraint_group, constraint_state)], misc=misc)

    def create_multiplier(self, multiplier_kwargs: dict):
        # Using an IndexedMultiplier since some mini-batches of data may not contain
        # some groups of data, in which case we can not compute a constraint for them.
        # Cooper's IndexedMultiplier handle this by allowing us to provide measurements
        # of the some constraints, and not others.
        return cooper.multipliers.IndexedMultiplier(
            constraint_type=self.constraint_type, num_constraints=self.num_constraints, **multiplier_kwargs
        )

    def dual_parameter_groups(self):
        return {"multipliers": self.multiplier.parameters()}
