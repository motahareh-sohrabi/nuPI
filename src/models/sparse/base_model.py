from abc import ABC, abstractmethod

from torch import nn

from .layers import BaseL0Layer, MaskedBatchNorm2d


class BaseL0Model(nn.Module, ABC):
    def __init__(self, input_shape: tuple[int], output_size: int, droprate_init: float = 0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_shape = input_shape
        self.output_size = output_size
        self.droprate_init = droprate_init

    def parameter_groups(self):
        gate_parameters = []
        weight_parameters = []
        for layer in self.modules():
            if isinstance(layer, BaseL0Layer):
                gate_parameters.append(layer.weight_log_alpha)
                weight_parameters.append(layer.weight)
                if layer.bias is not None:
                    weight_parameters.append(layer.bias)
                    if layer.bias_log_alpha is not None:
                        gate_parameters.append(layer.bias_log_alpha)
            elif isinstance(layer, MaskedBatchNorm2d) or isinstance(layer, nn.BatchNorm2d):
                weight_parameters.extend(layer.parameters())

        return {"params": weight_parameters, "gates": gate_parameters}

    @abstractmethod
    def forward(self, x):
        pass
