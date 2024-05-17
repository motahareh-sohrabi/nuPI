import dotenv
import torch
import wandb

import shared
from src import utils

from .base_trainer import BaseTrainer

# Load environment variables from .env file. This file is not tracked by git.
dotenv.load_dotenv()


logger = shared.fetch_main_logger(apply_basic_config=True)


class SVMTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

    def _train_one_epoch(self, train_data_iter):
        logger.info(f"Initiating training loop on rank {self.dist.rank}")

        self.model.train()

        train_meters = self._initialize_meters("train")

        while True:
            try:
                batch_data = next(train_data_iter)
            except StopIteration:
                if self.is_main_process:
                    train_metrics = {log_name: meter.avg for log_name, meter in train_meters.items()}
                    if self.steps_taken % self.config.logging.print_train_stats_period_steps == 0:
                        logger.info(f"Training metrics at epoch {self.epoch} (step {self.steps_taken}):")
                        for name, metric in train_metrics.items():
                            logger.info(f"\t* {name}: {train_metrics[name] :.4f}")

                    wandb.log(self._gather_log_metrics(train_metrics, prefix="train/"), step=self.steps_taken)

                    # Fresh iterator for next epoch
                    return iter(self.dataloaders.train)

            inputs = batch_data[0].to(device=self.device, non_blocking=True)
            targets = batch_data[1].to(device=self.device, non_blocking=True)

            if self.config.task.use_indexed_dataset:
                # If the training dataset is an IndexedDataset, the third element of the
                # batch_data tuple is the indices corresponding to the batch samples.
                data_indices = batch_data[2].to(device=self.device, non_blocking=True)
            else:
                data_indices = None

            constraint_features = data_indices

            compute_cmp_state_fn = lambda: self.cmp.compute_cmp_state(
                model=self.model, inputs=inputs, targets=targets, constraint_features=constraint_features
            )
            cmp_state, lagrangian_store = self.cooper_optimizer.roll(compute_cmp_state_fn=compute_cmp_state_fn)

            # Update meters
            for name, metric in self.metrics.train.items():
                predictions = cmp_state.misc.pop("predictions")
                metric_value = metric(predictions, targets, get_items=True)
                for key, value in metric_value.items():
                    train_meters[key].update(value, n=inputs.shape[0])

            if self.is_main_process:
                violations = cmp_state.observed_constraints[0][1].violation.detach()

                # ----------------------------- Logging --------------------------------
                multipliers = self.multiplier.weight.detach()

                violations_dict = utils.flatten_tensor_for_logging(
                    violations, indices=data_indices, prefix="constraints/violation"
                )
                multipliers_dict = utils.flatten_tensor_for_logging(multipliers, prefix="constraints/multipliers")

                # NOTE: logging the violations one-by-one *and* all together as a histogram
                train_log_dict = {
                    **cmp_state.misc,
                    **violations_dict,
                    **multipliers_dict,
                    "lagrangian": lagrangian_store.lagrangian.detach(),
                    "violations_hist": wandb.Histogram(violations.cpu()),
                    "multipliers_hist": wandb.Histogram(multipliers.cpu()),
                    "multiplier_balance": cmp_state.misc["multiplier_balance"].cpu(),
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

        val_meters = self._initialize_meters("val")

        with torch.inference_mode():
            for batch_data in self.dataloaders.val:
                inputs = batch_data[0].to(device=self.device, non_blocking=True)
                targets = batch_data[1].to(device=self.device, non_blocking=True)
                predictions = self.model(inputs)

                for name, metric in self.metrics.val.items():
                    metric_value = metric(predictions, targets, get_items=True)
                    for key, value in metric_value.items():
                        val_meters[key].update(value, n=inputs.shape[0])

        if self.is_main_process:
            logger.info(f"Aggregating validation metrics on rank {self.dist.rank}")
            val_metrics = {log_name: meter.avg for log_name, meter in val_meters.items()}
            logger.info(f"Validation metrics at epoch {self.epoch} (step {self.steps_taken}):")
            for name, metric in val_metrics.items():
                logger.info(f"\t* {name}: {val_metrics[name] :.4f}")

            val_log_dict = self._gather_log_metrics(val_metrics, prefix="val/")
            wandb.log(val_log_dict, step=self.steps_taken)

        logger.info(f"Finished measuring validation metrics")

    @torch.no_grad()
    def _log_after_training(self):
        multiplier_values = self.cmp.constraint_group.multiplier.weight.flatten().detach()
        logger.info("Number non-zero multipliers at end of training: %d", torch.sum(multiplier_values > 0))

        # Plot the decision boundary if the data is 2D
        if self.dataloaders.train.dataset.data.shape[1] == 2:
            plot_kwargs = dict(up_title="Decision Boundary", show=False, filename_prefix="", show_max_margin=True)
            decision_boundary = utils.plots.plot_2d_decision_boundary(
                self.model, self.dataloaders.train.dataset.data, self.dataloaders.train.dataset.targets, **plot_kwargs
            )

            wandb.log(
                {"decision_boundary": wandb.Image(decision_boundary), "_epoch": self.epoch}, step=self.steps_taken
            )

    def _should_terminate(self):
        return super()._should_terminate()
