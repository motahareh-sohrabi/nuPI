from typing import Type

import numpy as np
import torch
import torch.nn as nn

from .base_model import BaseModel


class MLP(BaseModel):
    def __init__(
        self,
        input_shape: tuple[int],
        output_size: int,
        hidden_sizes: tuple[int] = (),
        activation_type: Type[nn.Module] = nn.ReLU,
        bias: bool = True,
    ):
        super(MLP, self).__init__(input_shape=input_shape, output_size=output_size)
        self.input_size = np.prod(*input_shape)
        self.output_size = output_size
        self.hidden_sizes = hidden_sizes

        layers = []

        # Add hidden layers
        in_size = self.input_size
        for out_size in hidden_sizes:
            layers.append(nn.Linear(in_size, out_size, bias=bias))
            layers.append(activation_type())
            in_size = out_size

        # Add output layer
        layers.append(nn.Linear(in_size, output_size, bias=bias))

        self.model = nn.Sequential(*layers)

    def squared_l2_norm(self, include_biases: bool = True):
        squared_norm = 0.0
        for param_name, param in self.named_parameters():
            if "bias" not in param_name or include_biases:
                squared_norm += torch.sum(torch.square(param))

        return squared_norm

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.model(x)
        return x

    def parameter_groups(self):
        return {"params": self.parameters()}


class LinearModel(MLP):
    def __init__(self, input_shape: int, output_size: int, bias: bool = True):
        super(LinearModel, self).__init__(input_shape=input_shape, output_size=output_size, hidden_sizes=(), bias=bias)
