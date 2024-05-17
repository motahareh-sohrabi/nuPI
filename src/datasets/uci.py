import pandas as pd
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

import shared

logger = shared.fetch_main_logger()

from .utils import BaseDataset


# load Iris Flower Dataset with only two classes
class IrisFlower(BaseDataset):
    num_classes = 2

    def __init__(
        self,
        split: str,
        num_features: int = 4,
        make_linearly_separable: bool = False,
        val_proportion: float = 0.3,
        random_state: int = 0,
        dtype=torch.float32,
    ):
        """
        Args:
            split: "train" or "val"
            num_features: number of features to use
            make_linearly_separable (bool): if True, we remove the "virginica" class from
                the dataset to make it linearly separable
            val_proportion: proportion of data to use for validation
            random_state: random seed for train/val split
        """

        assert 1 <= num_features <= 4, "Iris Flower dataset has only 4 features. Provide a number between 1 and 4."
        assert split in ["train", "val"], "Split must be either 'train' or 'val'"

        self.num_features = num_features
        self.make_linearly_separable = make_linearly_separable

        data, targets = self._load_data()

        train_data, val_data, train_targets, val_targets = train_test_split(
            data, targets, train_size=1 - val_proportion, random_state=random_state, stratify=targets
        )

        data, targets = (train_data, train_targets) if split == "train" else (val_data, val_targets)
        self.data = torch.tensor(data, dtype=dtype)
        self.targets = torch.tensor(targets, dtype=torch.long)

        self.input_shape = (self.data.shape[1],)
        self.output_size = 1

        logger.info(
            f"Created IrisFlower dataset for split {split} -- {num_features} features and {len(self.data)} samples"
        )

    def _load_data(self):
        data = load_iris()

        if self.make_linearly_separable:
            banned_target = list(data["target_names"]).index("virginica")
            filter_mask = data["target"] != banned_target
            targets = data["target"][filter_mask]
            data = data["data"][filter_mask]
        else:
            data = data["data"]
            targets = data["target"]

        data = data[:, : self.num_features]
        normalized_data = (data - data.mean(axis=0)) / data.std(axis=0)

        return normalized_data, targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)
