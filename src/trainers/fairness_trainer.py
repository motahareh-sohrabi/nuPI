import dotenv
import numpy as np
import torch
import wandb

import shared
from src import utils

from .base_trainer import BaseTrainer

# Load environment variables from .env file. This file is not tracked by git.
dotenv.load_dotenv()


logger = shared.fetch_main_logger(apply_basic_config=True)


class FairnessTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def _update_meters(self, meters, predictions, targets, groups):

        num_groups = self.cmp.num_groups
        values, counts = np.unique(groups, return_counts=True)
        # Ensure that the counts correspond to their respective groups
        num_samples_per_group = torch.zeros(num_groups, device=predictions.device)
        for g_idx, g in enumerate(self.cmp.all_protected_groups):
            idx = np.where(values == g)[0]
            if idx.size > 0:
                num_samples_per_group[g_idx] = counts[idx].item()

        total_samples = num_samples_per_group.sum()
        assert total_samples == predictions.shape[0], "Number of samples per group should sum to the batch size"

        for name, metric in self.metrics.train.items():
            metric_value = metric(
                predictions, targets, group=groups, all_protected_groups=self.cmp.all_protected_groups, get_items=True
            )
            for key, value in metric_value.items():
                n = num_samples_per_group if value.numel() > 1 else total_samples
                value[n == 0] = 0  # Ensure that the meter is not updated for non-observed groups
                meters[key].update(value, n=n)

    def _extract_metrics(self, meters):
        metrics = {}
        for name, meter in meters.items():
            name = name.split("@")[0]
            value = meter.avg
            if value.numel() == 1:
                metrics[f"{name}"] = value
            else:
                new_flat_metrics = utils.flatten_tensor_for_logging(value, prefix=name)
                metrics = {**metrics, **new_flat_metrics}

        # Compute constraints at the epoch level
        if "group/pos_probability_0" in metrics:
            if "pos_probability" not in metrics:
                raise RuntimeError(
                    "A meter for the aggregate positive probability should be present to compute epoch level constraints"
                )

            for g in range(self.cmp.num_groups):
                metrics[f"constraints/violation_{g}"] = 100 * (
                    metrics[f"group/pos_probability_{g}"] - metrics["pos_probability"]
                )

            violations = torch.stack([metrics[f"constraints/violation_{g}"] for g in range(self.cmp.num_groups)])
            violations = violations.flatten()

            metrics["constraints/max_violation"] = torch.max(torch.abs(violations))
            metrics["constraints/l1_violation"] = torch.sum(torch.abs(violations))

        return metrics, violations

    def _train_one_epoch(self, train_data_iter):
        logger.info(f"Initiating training loop on rank {self.dist.rank}")

        self.model.train()

        while True:
            try:
                batch_data = next(train_data_iter)
            except StopIteration:
                if self.is_main_process:

                    for scheduler in self.schedulers.primal + self.schedulers.dual:
                        if scheduler is not None:
                            current_lr = scheduler.get_last_lr()[0]
                            wandb.log({"learning_rate": current_lr}, step=self.steps_taken)

                    # Fresh iterator for next epoch
                    return iter(self.dataloaders.train)

            inputs = batch_data[0].to(device=self.device, non_blocking=True)
            targets = batch_data[1].to(device=self.device, non_blocking=True)
            groups = batch_data[2]

            compute_cmp_state_fn = lambda: self.cmp.compute_cmp_state(
                model=self.model, inputs=inputs, targets=targets, groups=groups
            )
            cmp_state, lagrangian_store = self.cooper_optimizer.roll(compute_cmp_state_fn=compute_cmp_state_fn)

            if self.is_main_process:
                # ----------------------------- Logging --------------------------------
                violations = cmp_state.observed_constraints[0][1].violation.detach()
                observed_constraints = cmp_state.observed_constraints[0][1].constraint_features
                # This returns the value of the multipliers that were updated in the last step
                multiplier_values = lagrangian_store.primal_constraint_stores[0].multiplier_value.detach()

                # Only logging violations and multipliers for observed constraints
                violations_dict = utils.flatten_tensor_for_logging(
                    violations, indices=observed_constraints, prefix="constraints/violation"
                )
                multipliers_dict = utils.flatten_tensor_for_logging(
                    multiplier_values, indices=observed_constraints, prefix="constraints/multipliers"
                )

                # NOTE: logging violations and multipliers one-by-one *and* all together
                # as a histogram
                train_log_dict = {
                    **cmp_state.misc,
                    **violations_dict,
                    **multipliers_dict,
                    "lagrangian": lagrangian_store.lagrangian.detach(),
                    "violations_hist": wandb.Histogram(violations.cpu()),
                    "multipliers_hist": wandb.Histogram(multiplier_values.cpu()),
                }
                train_log_dict = self._gather_log_metrics(train_log_dict, prefix="train/batch/")

                wandb.log(train_log_dict, step=self.steps_taken)

            if self.steps_taken % self.config.logging.print_train_stats_period_steps == 0:
                logger.info(
                    f"Step {self.steps_taken}/{self.num_steps} | Epoch {self.epoch} | Lagrangian: {lagrangian_store.lagrangian:.4f}"
                )

            self.steps_taken += 1

    def _val_one_epoch(self):
        logger.info(f"Initiating validation loop on rank {self.dist.rank}")
        self.model.eval()

        train_meters = self._initialize_meters("train")
        val_meters = self._initialize_meters("val")

        with torch.inference_mode():

            for split in ["train", "val"]:
                for batch_data in getattr(self.dataloaders, split):
                    inputs = batch_data[0].to(device=self.device, non_blocking=True)
                    targets = batch_data[1].to(device=self.device, non_blocking=True)
                    groups = batch_data[2]

                    predictions = self.model(inputs)

                    split_meters = train_meters if split == "train" else val_meters
                    self._update_meters(split_meters, predictions, targets, groups)

                if self.is_main_process:
                    logger.info(f"Aggregating {split} metrics on rank {self.dist.rank}")

                    split_logs, violations = self._extract_metrics(train_meters if split == "train" else val_meters)

                    logger.info(f"Training metrics at epoch {self.epoch} (step {self.steps_taken}):")
                    for name, metric in split_logs.items():
                        logger.info(f"\t* {name}: {split_logs[name] :.4f}")

                    prefix = "train_eval/" if split == "train" else "val/"
                    train_log_dict = self._gather_log_metrics(split_logs, prefix=prefix)
                    wandb.log(train_log_dict, step=self.steps_taken)

                if split == "train":
                    # NOTE: we are performing an update on the multipliers based on
                    # violations measured on the entire train dataset.

                    # We divide by 100 since this quantity was previously multiplied by
                    # 100 for logging purposes
                    prepared_violations = violations.reshape(self.cmp.multiplier.weight.shape) / 100
                    self.cmp.multiplier.weight.grad = prepared_violations.to(self.cmp.multiplier.device)
                    self.cooper_optimizer.dual_step()

        logger.info(f"Finished evaluation on rank {self.dist.rank}")

    def _should_terminate(self):
        return super()._should_terminate()
