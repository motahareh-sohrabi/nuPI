import random
from types import SimpleNamespace
from typing import Callable

import numpy as np
import torch


def extract_to_namespace(struct: dict | SimpleNamespace, extract_fn: Callable):
    if isinstance(struct, dict):
        return SimpleNamespace(**{key: extract_fn(value) for (key, value) in struct.items()})
    elif isinstance(struct, SimpleNamespace):
        return SimpleNamespace(**{key: extract_fn(getattr(struct, key)) for key in vars(struct)})
    else:
        raise ValueError(f"Unsupported type {type(struct)}")


def ensure_iterable(x):
    if isinstance(x, (list, tuple)):
        return x
    elif isinstance(x, torch.Tensor):
        return x.flatten().tolist()
    else:
        return [x]


def flatten_tensor_for_logging(tensor: torch.Tensor, indices=None, prefix: str = ""):
    if indices is None:
        # Enumerate the tensor if no indices are provided
        return {f"{prefix}_{k}": v for k, v in enumerate(ensure_iterable(tensor))}
    else:
        # The entries in the tensor correspond to the provided indices
        assert len(indices) == len(tensor.flatten())
        return {f"{prefix}_{k}": v for k, v in zip(indices, ensure_iterable(tensor))}


def set_seed(seed: int):
    """Sets the seed for the random number generators used by random, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RNGContext:
    """Context manager for deterministic random number generation. This context returns
    to the previous RNG states after its closure."""

    def __init__(self, seed):
        """Initializes the RNGContext object with a seed value."""
        self.seed = seed
        self.initial_torch_seed = torch.initial_seed()
        self.initial_numpy_seed = np.random.get_state()
        self.initial_random_seed = random.getstate()

    def __enter__(self):
        """Sets the seed for the random number generators used by random, numpy and
        torch when entering the context."""
        set_seed(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        """Resets the RNG states to their previous values when exiting the context."""
        torch.manual_seed(self.initial_torch_seed)
        torch.cuda.manual_seed_all(self.initial_torch_seed)
        torch.random.manual_seed(self.initial_torch_seed)
        np.random.set_state(self.initial_numpy_seed)
        random.setstate(self.initial_random_seed)
