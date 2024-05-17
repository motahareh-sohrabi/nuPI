import logging
import operator
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

import cooper
import dotenv
import pydash
import submitit
import torch
import wandb
from torch.nn.parallel import DistributedDataParallel

import shared
from src import datasets, models, optim, utils

# Load environment variables from .env file. This file is not tracked by git.
dotenv.load_dotenv()

logger = shared.fetch_main_logger(apply_basic_config=True)


class BaseTrainer(ABC):
    def __init__(self, config):
        self.config = config

        logger.info("Initialized trainer with configuration:")
        logger.info(config)
        logger.info(f"Current working directory is {os.getcwd()}")

    def __call__(self):
        self._make_reproducible()

        self.dist = utils.distributed.init_distributed(self.config.resources)
        self.device = self.dist.device
        self.dtype = self.config.task.dtype
        self.is_main_process = self.dist.rank == 0

        # Update the trainer logger to include rank information for multi-GPU training
        self._update_logger()

        self.wandb_run, self.run_checkpoint_dir = self._create_wandb_logger()

        logger.info("Trainer called with config:")
        logger.info(self.config)

        self.datasets = self._create_datasets()
        self.dataloaders = self._create_dataloaders()
        # moose:remove
        # self.num_samples = utils.extract_to_namespace(self.datasets, extract_fn=lambda dataset: len(dataset))
        self.num_batches = utils.extract_to_namespace(self.dataloaders, extract_fn=lambda loader: len(loader))

        self.model = self._create_model().to(dtype=self.dtype)

        self.num_steps = self._init_stopping_condition()
        self.eval_period_steps = self._init_evaluation_period()

        self.metrics = self._create_metrics()

        self.cmp, self.multiplier = self._create_cmp_and_multiplier()
        self.cooper_optimizer, self.schedulers = self._create_optimizers_and_schedulers()

        self.train()

        self._log_after_training()
        self._clean_finish()

    def _update_logger(self):
        shared.configure_logger(
            logger=shared.fetch_main_logger(),
            custom_format=f"(Rank:{self.dist.rank}/WS:{self.dist.world_size}) %(module)s:%(funcName)s:%(lineno)d | %(message)s ",
            level=getattr(logging, self.config.logging.log_level),
            show_path=self.config.logging.wandb_mode == "disabled",  # Only show path hyperlinks if not using wandb
        )

    def _make_reproducible(self):
        utils.set_seed(self.config.train.seed)
        if self.config.train.use_deterministic_ops:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _create_datasets(self):
        dataset_dict = {}
        for split in self.config.data.dataset_kwargs.split_kwargs:
            split_kwargs = pydash.omit(self.config.data.dataset_kwargs, "split_kwargs")
            split_kwargs.update(getattr(self.config.data.dataset_kwargs.split_kwargs, split, {}))
            dataset_dict[split] = self.config.data.dataset_class(**split_kwargs)
            if hasattr(dataset_dict[split], "data"):
                dataset_dict[split].data = dataset_dict[split].data.to(dtype=self.dtype)

            # Wrap the dataset in an IndexedDataset if requested
            if self.config.task.use_indexed_dataset:
                dataset_dict[split] = datasets.IndexedDataset(dataset_dict[split])

        return SimpleNamespace(**dataset_dict)

    def _create_dataloaders(self):
        if self.config.data.dataloader_kwargs.use_distributed_sampler and not self.dist.multi_gpu:
            raise ValueError("Distributed sampler requires multi-gpu training.")

        best_num_workers = None
        dataloaders = {}
        for split in self.config.data.dataloader_kwargs.split_kwargs:
            split_kwargs = pydash.omit(self.config.data.dataloader_kwargs, ["split_kwargs", "num_workers"])
            split_kwargs.update(getattr(self.config.data.dataloader_kwargs.split_kwargs, split, {}))

            dataloader, last_num_workers = datasets.build_dataloader(
                dataset=getattr(self.datasets, split),
                split=split,
                device=self.device,
                num_workers=getattr(self.config.data.dataloader_kwargs, "num_workers", best_num_workers),
                **split_kwargs,
            )
            dataloaders[split] = dataloader

            # Use the number of workers from the first dataloader for the rest -- this
            # avoids having to find the best number of workers for each split
            best_num_workers = last_num_workers if best_num_workers is None else best_num_workers

            logger.info(
                f"Initialized {split} dataloader with batch size {dataloader.batch_size} and {dataloaders[split].num_workers} workers"
            )

        return SimpleNamespace(**dataloaders)

    def _create_model(self):
        logger.info("Starting model creation")
        init_kwargs = pydash.omit(self.config.model.init_kwargs, "from_trainer")
        with utils.RNGContext(self.config.model.init_seed):
            kwargs_from_trainer = {}
            for key, value in self.config.model.init_kwargs.from_trainer.items():
                kwargs_from_trainer[key] = operator.attrgetter(value)(self)

            model = self.config.model.model_class(
                input_shape=self.datasets.train.input_shape,
                output_size=self.datasets.train.output_size,
                **init_kwargs,
                **kwargs_from_trainer,
            )

        model.to(device=self.device)
        param_count = sum([torch.prod(torch.tensor(p.shape)).item() for p in model.parameters()])
        logger.info(f"Created model {self.config.model.model_class} with " + f"{param_count} parameters")

        if self.config.resources.use_ddp:
            resources_config = self.config.resources
            total_gpus = resources_config.gpus_per_task * resources_config.tasks_per_node * resources_config.nodes

            if total_gpus == 1:
                logger.warning("Requested using DDP but only 1 GPU available. Continuing without DDP.")
            else:
                model = DistributedDataParallel(model)
                logger.info("Successfully wrapped model in DDP.")

        return model

    def _create_cmp_and_multiplier(self):
        logger.info("Starting CMP and multiplier creation")
        cmp_kwargs = pydash.omit(self.config.task.cmp_kwargs, "from_trainer")
        # Extract attributes from trainer needed by the CMP constructor
        kwargs_from_trainer = {
            key: operator.attrgetter(value)(self) for key, value in self.config.task.cmp_kwargs.from_trainer.items()
        }

        multiplier_kwargs = self.config.task.multiplier_kwargs
        if "init" in multiplier_kwargs:
            init = torch.tensor(multiplier_kwargs["init"])
            if len(init.shape) == 0:
                init.unsqueeze_(0)
            multiplier_kwargs = pydash.omit(multiplier_kwargs, "init")
            multiplier_kwargs = {"device": self.device, "init": init, **multiplier_kwargs}
        else:
            multiplier_kwargs = {"device": self.device, **multiplier_kwargs}

        logger.info(f"Building {self.config.task.cmp_class.__name__} with kwargs: {cmp_kwargs}")

        cmp = self.config.task.cmp_class(multiplier_kwargs=multiplier_kwargs, **cmp_kwargs, **kwargs_from_trainer)
        if cmp.has_dual_variables:
            multiplier = cmp.multiplier.to(dtype=self.dtype)
        else:
            multiplier = None

        if self.config.resources.use_ddp and cmp.has_dual_variables:
            multiplier = DistributedDataParallel(multiplier)
            logger.info("Successfully wrapped multiplier in DDP.")

        return cmp, multiplier

    def _create_optimizers_and_schedulers(self):
        return optim.build_cooper_optimizer_and_schedulers(model=self.model, cmp=self.cmp, config=self.config)

    def _save_checkpoint(self):
        if self.is_main_process and self.config.checkpointing.enabled:
            os.makedirs(self.run_checkpoint_dir, exist_ok=True)

            checkpoint = {
                "model": self.model.state_dict(),
                "cooper_optimizer": self.cooper_optimizer.state_dict(),
                "steps_taken": self.steps_taken,
                "epoch": self.epoch,
                "elapsed_time": (time.time() - self.start_time) + self.elapsed_time,
            }

            for scheduler in self.schedulers.primal + self.schedulers.dual:
                if scheduler is not None:
                    checkpoint[f"{scheduler}_scheduler"] = scheduler.state_dict()

            torch.save(checkpoint, os.path.join(self.run_checkpoint_dir, "checkpoint.pt"))
            logger.info(f"Saved checkpoint to {self.run_checkpoint_dir} (step={self.steps_taken}; epoch={self.epoch})")

    def _load_checkpoint(self):
        logger.info("Attempting to resume from checkpoint...")
        if os.path.isfile(os.path.join(self.run_checkpoint_dir, "checkpoint.pt")):
            checkpoint = torch.load(os.path.join(self.run_checkpoint_dir, "checkpoint.pt"))

            self.steps_taken = checkpoint["steps_taken"]
            self.epoch = checkpoint["epoch"]
            self.elapsed_time = checkpoint["elapsed_time"]
            self.start_time = time.time()
            self.model.load_state_dict(checkpoint["model"])

            cooper_optimizer_state_dict = checkpoint["cooper_optimizer"]
            constrained_optimizer_kwargs = {
                "primal_optimizers": self.cooper_optimizer.primal_optimizers,
                "dual_optimizers": self.cooper_optimizer.dual_optimizers,
                "multipliers": self.cooper_optimizer.multipliers,
            }
            self.cooper_optimizer = cooper.optim.utils.load_cooper_optimizer_from_state_dict(
                cooper_optimizer_state=cooper_optimizer_state_dict, **constrained_optimizer_kwargs
            )

            for scheduler in self.schedulers.primal + self.schedulers.dual:
                if scheduler is not None:
                    scheduler.load_state_dict(checkpoint[f"{scheduler}_scheduler"])
        else:
            raise ValueError("WandB run requested resuming but no checkpoint found.")

    def _create_metrics(self):
        metrics = SimpleNamespace()
        for split in ["train", "val"]:
            metrics_in_split = {}
            for metric_config in getattr(self.config.metrics, f"{split}_metrics"):
                name = metric_config["name"]
                metric = metric_config["metric"]
                kwargs = metric_config["kwargs"] if "kwargs" in metric_config else {}
                metrics_in_split[name] = getattr(utils.metrics, metric)(**kwargs)
            metrics.__setattr__(split, metrics_in_split)
            logger.info(f"Instantiated {len(metrics_in_split)} {split} metrics")

        return metrics

    def _create_wandb_logger(self):
        run, run_checkpoint_dir = None, None
        if self.is_main_process:
            is_local_job = not ("SLURM_JOB_ID" in os.environ.keys() and self.config.resources.cluster == "slurm")

            # This is compatible with preemption since the SLURM_JOB_ID value is
            # preserved after preemption.
            custom_run_id = None if is_local_job else os.environ["SLURM_JOB_ID"]

            run = wandb.init(
                entity=os.environ["WANDB_ENTITY"],
                project=os.environ["WANDB_PROJECT"],
                dir=os.environ["WANDB_DIR"],
                id=custom_run_id,
                mode=self.config.logging.wandb_mode,
                resume="allow",
                tags=self.config.logging.wandb_tags,
            )
            logger.info(f"Initialized WandB run with id {run.id}")

            wandb.config.update(self.config.to_dict(), allow_val_change=True)

            # Define metrics for custom x-axis
            wandb.define_metric("_epoch")
            wandb.define_metric("val/*", step_metric="_epoch")

            run_subdir = run.id if is_local_job else os.environ["SLURM_JOB_ID"]
            run_checkpoint_dir = Path(os.environ["CHECKPOINT_DIR"]) / run_subdir

        return run, run_checkpoint_dir

    def _init_stopping_condition(self):
        train_config = self.config.train
        if train_config.total_epochs is not None and train_config.total_steps is not None:
            raise ValueError("Train config contains both 'total_epochs' and 'total_steps'. Please specift only one")
        elif train_config.total_steps is not None:
            num_steps = train_config.total_steps
        elif train_config.total_epochs is not None:
            num_steps = self.num_batches.train * train_config.total_epochs
        else:
            raise ValueError("No stopping condition was specified.")

        num_epochs = num_steps / self.num_batches.train
        logger.info(f"Training loop was configured to run for {num_steps} steps ({num_epochs: .2f} epochs)")

        return num_steps

    def _init_evaluation_period(self):
        eval_period_steps = self.config.logging.eval_period_steps
        eval_period_epochs = self.config.logging.eval_period_epochs

        if eval_period_steps is not None and eval_period_epochs is not None:
            raise ValueError("Train config should specify exactly one of 'eval_period_steps' and 'eval_period_epochs'.")
        if eval_period_steps:
            _eval_period_steps = eval_period_steps
        elif eval_period_epochs:
            _eval_period_steps = self.num_batches.train * eval_period_epochs
        else:
            raise ValueError("No evaluation period was specified.")

        _eval_period_epochs = _eval_period_steps / self.num_batches.train
        logger.info(f"Evaluation happening every {_eval_period_steps} steps ({_eval_period_epochs: .2f} epochs)")

        return _eval_period_steps

    def _gather_log_metrics(self, metrics: dict[str, float], prefix: str = "train/"):
        wandb_dict = {prefix + k: v for k, v in metrics.items()}
        wandb_dict["_epoch"] = self.epoch
        wandb_dict["wall_sec"] = time.time() - self.start_time
        wandb_dict["training_steps"] = self.steps_taken

        return wandb_dict

    def _initialize_meters(self, split: str):
        meters = {}
        for _, metric in getattr(self.metrics, split).items():
            for key in metric.known_returns:
                if key in meters:
                    raise ValueError(f"Metric {key} is returned by multiple metrics")
                meters[key] = utils.meters.AverageMeter()

        return meters

    def __submitit_checkpoint__(self):
        """Function used by submitit when SLURM job is preempted"""
        resume_config = self.config.copy()
        with resume_config.unlocked():
            resume_config.checkpointing.resume_from_checkpoint = True
        resume_trainer = self.__class__(resume_config)
        return submitit.helpers.DelayedSubmission(resume_trainer)

    def _clean_finish(self):
        utils.distributed.wait_for_all_processes()

        if self.is_main_process:
            logger.info("Attempting to close WandB logger")
            wandb.finish()
            logger.info("Shutting down gracefully")

    def train(self):
        if self.wandb_run.resumed or self.config.checkpointing.resume_from_checkpoint:
            # Retrieves self.{steps_taken, epoch, elapsed_time} and loads checkpointed
            # state_dicts for the model, optimizers and schedulers.
            self._load_checkpoint()
        else:
            self.steps_taken = 0
            self.epoch = 0
            self.elapsed_time = 0
            self.start_time = time.time()
            logger.info("No checkpoint found, starting from scratch.")

            self._save_checkpoint()

        steps_since_last_epoch = self.steps_taken % len(self.dataloaders.train)
        if self.config.data.dataloader_kwargs.use_distributed_sampler:
            self.dataloaders.train.sampler.set_epoch(self.epoch)

        # Skip the training dataloader ahead to the current step
        train_data_iter = iter(self.dataloaders.train)
        for _ in range(steps_since_last_epoch):
            batch_data = next(train_data_iter)

        # After loading a checkpoint, and forwarding the dataloader and schedulers,
        # we are ready to train.
        self._train_loop(train_data_iter)

        logger.info("Training complete")

    def _train_loop(self, train_data_iter):
        logger.info("Starting training")

        logger.info(f"Evaluating validation performance at epoch {self.epoch} (step {self.steps_taken})")
        self._val_one_epoch()

        self.model.train()
        self._save_checkpoint()
        logger.info(f"Validation loop completed after step {self.steps_taken}")

        while True:
            logger.info(f"Starting epoch {self.epoch}")
            self.model.train()
            train_data_iter = self._train_one_epoch(train_data_iter)

            for scheduler in self.schedulers.primal + self.schedulers.dual:
                if scheduler is not None:
                    logger.info(f"Stepping {scheduler} at epoch {self.epoch}")
                    logger.info(f"Current LR: {scheduler.get_last_lr()}")
                    scheduler.step()

            self._save_checkpoint()
            logger.info(f"Finished epoch {self.epoch} after step {self.steps_taken}")
            self.epoch += 1

            if self.config.data.dataloader_kwargs.use_distributed_sampler:
                self.dataloaders.train.sampler.set_epoch(self.epoch)

            if self.steps_taken % self.eval_period_steps == 0:
                logger.info(f"Evaluating validation performance at epoch {self.epoch} (step {self.steps_taken})")
                self.model.eval()
                self._val_one_epoch()
                logger.info(f"Validation loop completed at epoch {self.epoch} (step {self.steps_taken})")

            self._save_checkpoint()

            self.elapsed_time = time.time() - self.start_time
            logger.info(f"Completed {self.steps_taken} steps of training" + f" ({self.elapsed_time:.2f} seconds)")

            # Wait for all processes to finish before final validation
            utils.distributed.wait_for_all_processes()

            if self._should_terminate():
                break

        # Wait for all processes to finish
        utils.distributed.wait_for_all_processes()

        # Final eval after training if we didn't just do one
        if not self.steps_taken % self.eval_period_steps == 0:
            logger.info("Final model evaluation")
            self._val_one_epoch()
            self._save_checkpoint()

    @abstractmethod
    def _train_one_epoch(self, train_data_iter):
        raise NotImplementedError

    @abstractmethod
    def _val_one_epoch(self):
        raise NotImplementedError

    @abstractmethod
    def _should_terminate(self):
        # NOTE: this method is called at the end of each epoch in _train_loop to stop
        # the training loop. By default, termination happens after a certain number of
        # steps. Just to ensure this method is used correctly, it is abstract and must
        # be implemented by subclasses.
        if self.steps_taken >= self.num_steps:
            logger.info("Stopping training: reached maximum number of steps!")
            return True

        return False

    def _log_after_training(self):
        pass
