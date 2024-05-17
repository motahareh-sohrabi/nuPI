import abc

import torch.nn as nn


class BaseModel(nn.Module, abc.ABC):
    def __init__(self, input_shape: tuple[int], output_size: int, *args, **kwargs):
        super(BaseModel, self).__init__()

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    @abc.abstractmethod
    def squared_l2_norm(self):
        pass

    @abc.abstractmethod
    def parameter_groups(self):
        """Defines parameter groups to be used when constructing the optimizer."""
        raise NotImplementedError
