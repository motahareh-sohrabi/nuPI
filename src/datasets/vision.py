import os

import torchvision as tv

import shared

logger = shared.fetch_main_logger()

from .utils import BaseDataset


class CIFAR10(BaseDataset):
    train_transforms = tv.transforms.Compose(
        [
            tv.transforms.RandomCrop(32, padding=4),
            tv.transforms.RandomHorizontalFlip(),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    val_transforms = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ]
    )
    input_shape = (3, 32, 32)
    output_size = 10

    def __init__(self, data_path: str, split: str):
        is_train = split == "train"
        transforms = self.train_transforms if is_train else self.val_transforms

        if data_path is None:
            data_path = os.path.join(os.environ["SLURM_TMPDIR"], "cifar10")

        self.dataset = tv.datasets.CIFAR10(root=data_path, train=is_train, transform=transforms, download=True)
        logger.info(f"CIFAR10 dataset {split} split contains {len(self.dataset)} samples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)
