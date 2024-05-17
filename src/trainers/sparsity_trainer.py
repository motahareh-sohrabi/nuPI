import dotenv
import torch
import wandb

import shared
from src import utils

from .base_trainer import BaseTrainer

# Load environment variables from .env file. This file is not tracked by git.
dotenv.load_dotenv()


logger = shared.fetch_main_logger(apply_basic_config=True)


class SparsityTrainer(BaseTrainer):
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

                    for scheduler in self.schedulers.primal + self.schedulers.dual:
                        if scheduler is not None:
                            current_lr = scheduler.get_last_lr()[0]
                            train_metrics[f"learning_rate"] = current_lr

                    wandb.log(self._gather_log_metrics(train_metrics, prefix="train/"), step=self.steps_taken)

                    # Fresh iterator for next epoch
                    return iter(self.dataloaders.train)

            inputs = batch_data[0].to(device=self.device, non_blocking=True)
            targets = batch_data[1].to(device=self.device, non_blocking=True)

            compute_cmp_state_fn = lambda: self.cmp.compute_cmp_state(model=self.model, inputs=inputs, targets=targets)
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
                    violations, indices=None, prefix="constraints/violation"
                )
                multipliers_dict = utils.flatten_tensor_for_logging(multipliers, prefix="constraints/multipliers")

                if "layer_densities" in cmp_state.misc:
                    layer_exp_l0 = cmp_state.misc.pop("layer_densities")
                    layer_exp_l0_dict = utils.flatten_tensor_for_logging(layer_exp_l0, prefix="expected_l0/layer")
                    cmp_state.misc.update(layer_exp_l0_dict)

                    model_exp_l0 = cmp_state.misc.pop("model_density")
                    cmp_state.misc["expected_l0/model"] = model_exp_l0

                # NOTE: logging the violations one-by-one *and* all together as a histogram
                train_log_dict = {
                    **cmp_state.misc,
                    **violations_dict,
                    **multipliers_dict,
                    "max_violation": violations.max(),
                    "min_violation": violations.min(),
                    "num_violations": (violations > 0).sum(),
                    "lagrangian": lagrangian_store.lagrangian.detach(),
                    "violations_hist": wandb.Histogram(violations.cpu()),
                    "multipliers_hist": wandb.Histogram(multipliers.cpu()),
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

            # Get the sparsity metrics of the model in eval mode
            model_density, layer_densities, sq_l2_norm = self.cmp.compute_sparsity_stats(self.model, is_test_time=True)
            val_metrics["density/model"] = model_density
            val_metrics["sq_l2_norm"] = sq_l2_norm
            layer_densities_dict = utils.flatten_tensor_for_logging(layer_densities, prefix="density/layer")
            val_metrics.update(layer_densities_dict)

            val_log_dict = self._gather_log_metrics(val_metrics, prefix="val/")
            wandb.log(val_log_dict, step=self.steps_taken)

        logger.info(f"Finished measuring validation metrics")

    def _should_terminate(self):
        return super()._should_terminate()
