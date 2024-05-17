import cooper
import torch
from sklearn import svm

import shared
from src.datasets import IndexedDataset

from .base import BaseProblem

logger = shared.fetch_main_logger()


class MaxMarginClassification(BaseProblem):
    """NOTE: This class assumes a binary classification task with {0, 1} labels!"""

    has_dual_variables = True

    def __init__(
        self,
        train_dataset,
        weight_decay: float = 1.0,
        lmbda_tolerance: float = 1e-6,
        multiplier_kwargs: dict = {},
        ensure_LICQ: bool = False,
    ):
        logger.info(f"Setting up MaxMarginClassification problem with weight_decay={weight_decay}")

        if weight_decay < 0:
            raise ValueError(f"Weight decay must be non-negative, got {weight_decay}")
        if train_dataset.targets.dtype != torch.long:
            raise ValueError(f"Targets must be of type torch.long, got {train_dataset.targets.dtype}")

        self.weight_decay = weight_decay
        self.lmbda_tolerance = lmbda_tolerance

        self.constraint_type = cooper.ConstraintType.INEQUALITY
        self.num_constraints, self.multiplier = self.create_multiplier(train_dataset, multiplier_kwargs)
        self.constraint_group = cooper.ConstraintGroup(
            constraint_type=self.constraint_type,
            formulation_type=cooper.FormulationType.LAGRANGIAN,
            multiplier=self.multiplier,
        )

        logger.info(f"Training max-margin SVM with linear kernel on {train_dataset.data.shape[0]} samples")

        # Selecting a large C to attain a hard-margin SVM
        self.max_margin_classifier = svm.SVC(C=1e8, kernel="linear", gamma=0.0, random_state=0, tol=1e-8)
        self.max_margin_classifier.fit(X=train_dataset.data, y=train_dataset.targets)

        # We take the absolute value of the coeffs from sklearn since they are multiplied by the label
        self.lmbda_star = torch.zeros(train_dataset.data.shape[0])
        tensor_lmbda_star = torch.tensor(self.max_margin_classifier.dual_coef_).float().abs()
        self.lmbda_star[self.max_margin_classifier.support_] = tensor_lmbda_star
        self.lmbda_star = self.lmbda_star.to(self.multiplier.weight.device)
        logger.info(f"Number non-zero multipliers of max-margin SVM solution {torch.sum(self.lmbda_star > 0)}")
        if self.lmbda_star.numel() <= 10:
            logger.info(f"lmbda_star={self.lmbda_star}")

        if ensure_LICQ:
            support_indices = (self.lmbda_star > 0).cpu()
            support_vector_data = train_dataset.data[support_indices]
            if torch.linalg.matrix_rank(support_vector_data) != support_vector_data.shape[0]:
                raise ValueError("Support vectors are not linearly independent. Multipliers may not be unique.")

    def create_multiplier(self, train_dataset, multiplier_kwargs):
        num_constraints = train_dataset.data.shape[0]
        multiplier_class_name = "IndexedMultiplier" if isinstance(train_dataset, IndexedDataset) else "DenseMultiplier"
        multiplier_class = getattr(cooper.multipliers, multiplier_class_name)
        multiplier = multiplier_class(
            constraint_type=self.constraint_type, num_constraints=num_constraints, **multiplier_kwargs
        )
        return num_constraints, multiplier

    def compute_cmp_state(self, model, inputs, targets, constraint_features) -> cooper.CMPState:
        logits = model(inputs)
        pm_one_targets = (2 * targets) - 1
        per_sample_violation = 1 - logits.flatten() * pm_one_targets

        loss = model.squared_l2_norm(include_biases=False) * self.weight_decay / 2 if self.weight_decay != 0 else None
        avg_accuracy = (logits > 0).flatten().eq(targets).float().mean()

        const_state = cooper.ConstraintState(violation=per_sample_violation, constraint_features=constraint_features)

        multiplier_values = self.constraint_group.multiplier.weight.flatten().detach()
        dist2lmbda_star = torch.linalg.vector_norm(self.lmbda_star - multiplier_values, ord=2) ** 2

        # Need to re-arrange the multiplier values with the same order the samples
        multiplier_balance = torch.sum(multiplier_values[constraint_features] * pm_one_targets).abs().detach()

        batch_log_metrics = dict(
            loss=loss,
            avg_acc=avg_accuracy,
            max_violation=per_sample_violation.max(),
            multiplier_balance=multiplier_balance,
            dist2lmbda_star=dist2lmbda_star,
            predictions=logits,
        )

        return cooper.CMPState(
            loss=loss, observed_constraints=[(self.constraint_group, const_state)], misc=batch_log_metrics
        )

    def dual_parameter_groups(self):
        return {"multipliers": self.multiplier.parameters()}

    def check_stop_on_tolerance(self) -> bool:
        multiplier_values = self.multiplier.weight.flatten().detach()
        dist2lmbda_star = torch.linalg.vector_norm(self.lmbda_star - multiplier_values)
        logger.info(f"Current dist2lmbda_star={dist2lmbda_star}")

        return dist2lmbda_star < self.lmbda_tolerance
