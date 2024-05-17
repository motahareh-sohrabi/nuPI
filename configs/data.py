from functools import partial
from typing import Literal

import ml_collections as mlc

import shared
from src import datasets

MLC_PH = mlc.config_dict.config_dict.placeholder


def basic_config(splits: list[str] = ["train", "val"]):
    config = mlc.ConfigDict()

    config.dataset_class = MLC_PH(type)

    config.dataloader_kwargs = mlc.ConfigDict()
    config.dataloader_kwargs.split_kwargs = mlc.ConfigDict()
    config.dataloader_kwargs.use_distributed_sampler = MLC_PH(bool)
    config.dataloader_kwargs.seed = MLC_PH(int)
    config.dataloader_kwargs.use_prefetcher = MLC_PH(bool)
    config.dataloader_kwargs.num_workers = MLC_PH(int)

    config.dataset_kwargs = mlc.ConfigDict()
    config.dataset_kwargs.split_kwargs = mlc.ConfigDict()

    # Automatically create fields (unpopulated) for each of the splits
    for split_name in splits:
        setattr(config.dataset_kwargs.split_kwargs, split_name, mlc.ConfigDict())
        setattr(config.dataloader_kwargs.split_kwargs, split_name, mlc.ConfigDict())
        getattr(config.dataloader_kwargs.split_kwargs, split_name).batch_size_per_gpu = MLC_PH(int)

    return config


def iris():
    config = basic_config(splits=["train", "val"])

    config.dataset_class = datasets.IrisFlower

    config.dataloader_kwargs.use_distributed_sampler = False
    config.dataloader_kwargs.seed = 0
    config.dataloader_kwargs.use_prefetcher = False
    config.dataloader_kwargs.num_workers = 0
    config.dataloader_kwargs.split_kwargs.train.batch_size_per_gpu = -1
    config.dataloader_kwargs.split_kwargs.val.batch_size_per_gpu = -1

    config.dataset_kwargs.num_features = 4
    config.dataset_kwargs.make_linearly_separable = True
    config.dataset_kwargs.val_proportion = 0.3
    config.dataset_kwargs.random_state = 0
    config.dataset_kwargs.split_kwargs.train.split = "train"
    config.dataset_kwargs.split_kwargs.val.split = "val"

    return config


def adult():
    config = basic_config(splits=["train", "val"])

    config.dataset_class = datasets.Adult

    config.dataloader_kwargs.use_distributed_sampler = False
    config.dataloader_kwargs.seed = 0
    config.dataloader_kwargs.use_prefetcher = False

    config.dataloader_kwargs.split_kwargs.train.batch_size_per_gpu = 128
    config.dataloader_kwargs.split_kwargs.val.batch_size_per_gpu = 128

    config.dataset_kwargs.split_kwargs.train.split = "train"
    config.dataset_kwargs.split_kwargs.val.split = "val"
    config.dataset_kwargs.sensitive_attrs = ["sex", "race"]

    return config


def cifar():
    config = basic_config(splits=["train", "val"])

    config.dataset_class = datasets.CIFAR10

    config.dataloader_kwargs.use_distributed_sampler = False
    config.dataloader_kwargs.seed = 0
    config.dataloader_kwargs.use_prefetcher = False

    config.dataloader_kwargs.split_kwargs.train.batch_size_per_gpu = 128
    config.dataloader_kwargs.split_kwargs.val.batch_size_per_gpu = 128

    config.dataset_kwargs.split_kwargs.train.split = "train"
    config.dataset_kwargs.split_kwargs.train.data_path = None

    config.dataset_kwargs.split_kwargs.val.split = "val"
    config.dataset_kwargs.split_kwargs.val.data_path = None

    return config


DATA_CONFIGS = {"iris": iris, "adult": adult, "cifar10": cifar}


def get_config(config_string=None):
    return shared.default_get_config(
        config_key="data", pop_key="dataset_name", preset_configs=DATA_CONFIGS, cli_args=config_string
    )
