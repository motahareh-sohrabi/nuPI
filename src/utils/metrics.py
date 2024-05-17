import abc
from typing import Iterable

import numpy as np
import torch

MetricReturn = dict[str, torch.Tensor]


class Metric(abc.ABC):
    """Abstract class specifying the Metric interface"""

    def __init__(self, per_sample: bool = False, per_group: bool = False):

        if per_sample and per_group:
            raise ValueError("Metric can only be reported per sample or per group, not both.")

        self.per_sample = per_sample
        self.per_group = per_group

    @abc.abstractmethod
    def forward(self, prediction, target, group, all_protected_groups) -> MetricReturn:
        pass

    def __call__(self, prediction, target, group=None, all_protected_groups=None, get_items=False) -> MetricReturn:
        result = self.forward(prediction, target, group, all_protected_groups)
        if get_items:
            result = self.get_items(result)
        return result

    def get_items(self, result):
        extracted_result = {}
        for k, v in result.items():
            if isinstance(v, torch.Tensor):
                extracted_result[k] = v.detach()
            else:
                extracted_result[k] = v
        return extracted_result


def l2_loss(prediction: torch.Tensor, target: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
    per_sample_sq_loss = torch.linalg.vector_norm(prediction - target, ord=2, dim=1) ** 2
    return per_sample_sq_loss if per_sample else torch.mean(per_sample_sq_loss)


class L2Loss(Metric):
    known_returns = ["l2_loss"]

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, group, all_protected_groups) -> MetricReturn:
        return dict(l2_loss=l2_loss(prediction, target, per_sample=self.per_sample))


def cross_entropy(prediction: torch.Tensor, target: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
    reduction = "none" if per_sample else "mean"
    return torch.nn.functional.cross_entropy(prediction, target, reduction=reduction)


class CrossEntropy(Metric):
    known_returns = ["ce_loss"]

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, group, all_protected_groups) -> MetricReturn:
        return dict(ce_loss=cross_entropy(prediction, target, per_sample=self.per_sample))


def top1_accuracy(prediction: torch.Tensor, target: torch.Tensor, per_sample: bool = False) -> torch.Tensor:
    predicted_targets = torch.argmax(prediction, dim=1)
    per_sample_correct = (predicted_targets == target).float()
    return per_sample_correct if per_sample else per_sample_correct.sum() / prediction.shape[0]


class Accuracy(Metric):
    def __init__(self, which_k: Iterable[int] = (1,), per_sample: bool = False, per_group: bool = False):
        assert len(which_k) >= 1
        for val in which_k:
            assert val > 0
        self.which_k = which_k
        self.maxk: int = max(which_k)

        super().__init__(per_sample=per_sample, per_group=per_group)

        if per_sample:
            self.known_returns = [f"sample/acc@{k}" for k in which_k]
        elif per_group:
            self.known_returns = [f"group/acc@{k}" for k in which_k]
        else:
            self.known_returns = [f"acc@{k}" for k in which_k]

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor, group: list = None, all_protected_groups: list = None
    ) -> MetricReturn:
        with torch.no_grad():
            if prediction.shape[1] < self.maxk:
                raise ValueError(f"Number of classes ({prediction.shape[1]}) is less than maxk ({self.maxk}).")

            batch_size = target.size(0)
            result = {}
            if prediction.shape[1] == 1:
                correct = (prediction > 0).eq(target.view(-1, 1))
            else:
                _, pred = prediction.topk(self.maxk, 1, True, True)
                correct = pred.eq(target.view(-1, 1).expand_as(pred))

            if self.per_sample:
                for k in self.which_k:
                    any_correct_upto_k = torch.any(correct[..., :k], dim=1).float()
                    result[f"sample/acc@{k}"] = any_correct_upto_k

            elif self.per_group:
                group = np.array(group)
                for k in self.which_k:
                    any_correct_upto_k = torch.any(correct[..., :k], dim=1).float()

                    num_groups = len(all_protected_groups)
                    accuracy_k = torch.zeros(num_groups, device=prediction.device)

                    for g_idx in range(num_groups):
                        group_mask = all_protected_groups[g_idx] == group
                        accuracy_k[g_idx] = any_correct_upto_k[group_mask].mean()

                    result[f"group/acc@{k}"] = accuracy_k

            else:
                for k in self.which_k:
                    any_correct_upto_k = torch.any(correct[..., :k], dim=1).float()
                    acc_at_k = any_correct_upto_k.sum() / batch_size
                    result[f"acc@{k}"] = acc_at_k

            return result


class PositiveProbability(Metric):
    """
    Compute the mean probability of a positive class (i.e. the mean of the sigmoid of
    the predictions).

    Only supports one-dimensional inputs, as provided by a binary classification model.
    """

    def __init__(self, per_sample: bool = False, per_group: bool = False):
        super().__init__(per_sample=per_sample, per_group=per_group)

        if per_sample:
            self.known_returns = ["sample/pos_probability"]
        elif per_group:
            self.known_returns = ["group/pos_probability"]
        else:
            self.known_returns = ["pos_probability"]

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        group: torch.Tensor = None,
        all_protected_groups: list = None,
    ) -> MetricReturn:

        if prediction.shape[1] != 1:
            raise ValueError("PositiveProbability only supports binary classification models.")

        probs = torch.sigmoid(prediction)

        if self.per_sample:
            return {"sample/pos_probability": probs}
        elif self.per_group:
            group = np.array(group)

            num_groups = len(all_protected_groups)
            group_probs = torch.zeros(num_groups, device=prediction.device)
            for g_idx in range(num_groups):
                group_mask = all_protected_groups[g_idx] == group
                group_probs[g_idx] = probs[group_mask].mean()
            return {"group/pos_probability": group_probs}
        else:
            return {"pos_probability": probs.mean()}
