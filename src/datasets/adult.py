import os
import urllib.request

import numpy as np
import torch

import shared

logger = shared.fetch_main_logger()

from .utils import BaseDataset


class Adult(BaseDataset):
    def __init__(self, split: str, sensitive_attrs: list):
        self.split = split
        self.sensitive_attrs = sensitive_attrs

        self._load_data()
        self.input_shape = (self.data.shape[1],)
        self.output_size = 1
        logger.info(f"Created Adult dataset for split {split} -- {len(self.data)} samples")

    def _load_data(self):
        logger.info("Preprocessing data...")
        X_train, y_train, cols_to_normalize_train, protected_groups_train = preprocess_adult_data(
            split="train", sensitive_attrs=self.sensitive_attrs
        )
        norm_mean = X_train[:, cols_to_normalize_train].mean(axis=0)
        norm_std = X_train[:, cols_to_normalize_train].std(axis=0)

        if self.split == "train":
            X_train[:, cols_to_normalize_train] = (X_train[:, cols_to_normalize_train] - norm_mean) / norm_std
            data, targets, protected_groups = X_train, y_train, protected_groups_train
        elif self.split == "val":
            X_val, y_val, cols_to_normalize_val, protected_groups_val = preprocess_adult_data(
                split="val", sensitive_attrs=self.sensitive_attrs
            )
            assert cols_to_normalize_train == cols_to_normalize_val, "Columns to normalize must match for train and val"
            X_val[:, cols_to_normalize_train] = (X_val[:, cols_to_normalize_train] - norm_mean) / norm_std
            data, targets, protected_groups = X_val, y_val, protected_groups_val
            assert set(protected_groups_val).issubset(set(protected_groups_train)), "Unseen groups found in val split!"

        self.data = torch.tensor(data, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.long)
        self.protected_groups = protected_groups
        self.all_protected_groups = np.sort(np.unique(protected_groups_train))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index], self.protected_groups[index]


def preprocess_adult_data(split: str, sensitive_attrs=["sex", "race"], load_data_size=None):
    """
    Preprocesses the Adult dataset for training a fair classification model.

    Args:
        split (str): Specifies whether to load the train or test split of the dataset.
        load_data_size (Optional[int]): Number of samples to load from the dataset.

    Returns:
        Tuple: A tuple containing the following elements:
            - X (np.ndarray): The input features.
            - y (np.ndarray): The target labels.
            - x_control (dict): A dictionary containing the sensitive attributes.

    This function is adapted from the Fair Classification repository by Muhammad Bilal Zafar.
    https://github.com/mbilalzafar/fair-classification/blob/45e85baad209134bbe2697c1a884dd1e2c2d9786/disparate_impact/adult_data_demo/prepare_adult_data.py
    """

    if split not in ["train", "val"]:
        raise ValueError(f"Invalid split value: {split}")
    data_file = f"data/adult.{split}"

    if not os.path.exists(data_file):
        suffix = "data" if split == "train" else "test"
        url = f"https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.{suffix}"
        urllib.request.urlretrieve(url, data_file)

    all_attrs = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
    ]
    int_attrs = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
    attrs_to_ignore = ["sex", "race", "fnlwgt"]
    attrs_for_classification = sorted(list(set(all_attrs) - set(attrs_to_ignore) - set(sensitive_attrs)))

    x_control = {attr: [] for attr in sensitive_attrs}
    data_dict = {attr: [] for attr in attrs_for_classification}

    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file '{data_file}' not found. Please make sure the data files exist.")

    def clean_value(attr_name, value):
        if attr_name == "native_country":
            return "US" if value == "United-States" else "Non-US"
        elif attr_name == "education":
            if value in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                return "Prim-Middle-school"
            elif value in ["9th", "10th", "11th", "12th"]:
                return "High-school"
            return value

        return value

    targets = []
    for line in open(data_file, "r"):
        line = line.strip()

        if line == "":
            # Ignore empty lines
            continue
        line = line.split(", ")
        if len(line) != 15 or "?" in line:
            # We remove samples with missing values (marked as "?")
            continue

        for col_num in range(0, len(line) - 1):
            attr_name = all_attrs[col_num]
            attr_val = line[col_num]
            attr_val = clean_value(attr_name, attr_val)

            if attr_name in sensitive_attrs:
                # Sensitive attributes are not used for classification
                x_control[attr_name].append(attr_val)
            elif attr_name not in attrs_to_ignore:
                # If attribute is not sensitive and not ignored, we use it as a feature
                data_dict[attr_name].append(attr_val)

        raw_class_label = line[-1]
        if "50K" not in raw_class_label:
            raise Exception("Invalid class label value")
        class_label = 1 if ">50K" in raw_class_label else 0
        targets.append(class_label)

    for attr_name, attr_vals in data_dict.items():
        if attr_name not in int_attrs:
            # No need to convert attributes that already have integer values
            unique_vals = sorted(list(set(attr_vals)))
            integer_map = {val: i for i, val in enumerate(unique_vals)}
            data_dict[attr_name] = [integer_map[val] for val in attr_vals]

    X = []
    clean_feature_names = []
    for attr_name, attr_vals in data_dict.items():
        if attr_name in int_attrs + ["native_country"]:
            # Native country is already converted to binary 0/1
            X.append(np.array(attr_vals, dtype=np.float, ndmin=2).T)
            clean_feature_names.append(attr_name)
        else:
            one_hot_attr_vals, num_columns = convert_to_one_hot(attr_vals)
            clean_feature_names.extend([f"{attr_name}_{i}" for i in range(num_columns)])
            X.append(one_hot_attr_vals)
    logger.info(f"Feature names: {clean_feature_names}")

    # Stack together all the features
    X = np.concatenate(X, axis=1)
    targets = np.array(targets, dtype=int)

    # Concatenate group information. Ex: if a point belongs to Black and Female, its
    # entry in the protected_group array will be "BlackFemale"
    protected_group = np.array(X.shape[0] * [""])
    for k, v in x_control.items():
        protected_group = np.char.add(protected_group, np.array(v))

    # Count how many times each group appears
    unique_groups, group_counts = np.unique(protected_group, return_counts=True)
    logger.info(f"Split: {split}, Protected groups: {dict(zip(unique_groups, group_counts))}")

    if load_data_size is not None:
        print(f"Loading only {load_data_size} examples from the data")
        X = X[:load_data_size]
        targets = targets[:load_data_size]
        protected_group = protected_group[:load_data_size]

    # Columns containing numerical (i.e. non-one-hot) features that need to be normalized
    attrs_to_normalize = sorted(list(set(int_attrs).intersection(clean_feature_names)))
    cols_to_normalize = [clean_feature_names.index(attr) for attr in attrs_to_normalize]
    logger.info(f"Attributes to normalize: {attrs_to_normalize}")
    logger.info(f"Columns to normalize: {cols_to_normalize}")

    return X, targets, cols_to_normalize, protected_group


def convert_to_one_hot(input_array):
    num_columns = max(input_array) + 1
    out = np.zeros((len(input_array), num_columns))
    out[np.arange(len(input_array)), input_array] = 1
    return out, num_columns
