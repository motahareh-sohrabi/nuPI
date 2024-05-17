from dataclasses import dataclass, field
from typing import Optional, Union

import torch

import src.models.sparse.layers as layers


@dataclass
class ModelStats:
    """
    Model-level statistics, including sparsity and L2 regularization.

    Args:
        layer_stats: Dict containing a `LayerStats` object for each of the
            layers in the model.
        num_params: Total number of trainable parameters in the model. This
            counts both the weights and the biases.
        num_active_params: Number of active parameters in the model for the
            current masks. This count matches `num_params` for models without
            any MaskedLayers.
        num_sparse_params: Number of sparsifiable parameters in the model layer.
            This only counts parameters that can be removed through
            sparsification: for example, by applying magnitude pruning, or via
            trainable gates.
        num_active_sparse_params: Number of active sparsifiable parameters in
            the model. In the case of models with trainable gates, this measures
            the _expected_ number of active parameters.
        sq_l2_norm: Squared L2 norm of the model's parameters. This counts the
            squared L2 norm for both weights and biases. In the case of models
            with trainable gates, this measures the _expected_ squared L2 norm.
            The L2 norm of BatchNorm layers is not included in this computation
            (set to 0.0).
        num_gate_params: Total number of trainable parameters associated with
            the gates of different layers of the models. For models with no
            gated layers this is `None`.
    """

    layer_stats: dict[str, layers.LayerStats] = field(repr=False)

    num_params: int
    num_active_params: Union[int, torch.Tensor]

    num_sparse_params: int
    num_active_sparse_params: Union[int, torch.Tensor]

    sq_l2_norm: Optional[Union[float, torch.Tensor]] = None

    num_gate_params: Optional[int] = None

    def compute_density_stats(
        self,
    ) -> tuple[Union[float, torch.Tensor], Optional[Union[float, torch.Tensor]]]:
        """
        Compute density statistics for a layer.
        """

        density = self.num_active_params / self.num_params

        if self.num_sparse_params > 0:
            sparse_density = self.num_active_sparse_params / self.num_sparse_params
        else:
            sparse_density = None

        return density, sparse_density


def get_model_stats(model, is_test_time: bool = False) -> ModelStats:
    model_num_active_sparse_params = 0
    model_num_sparse_params = 0
    model_num_active_params = 0
    model_num_params = 0

    model_sq_l2_norm = 0.0
    layer_stats_list = []
    for layer in model.modules():
        if isinstance(layer, layers.BaseL0Layer):
            layer_stats = layer.layer_stats(is_test_time=is_test_time)
            layer_stats_list.append(layer_stats)

            # Model measurements
            model_num_active_sparse_params += layer_stats.num_active_sparse_params
            model_num_sparse_params += layer_stats.num_sparse_params
            model_num_active_params += layer_stats.num_active_params
            model_num_params += layer_stats.num_params

            model_sq_l2_norm += layer_stats.sq_l2_norm
        elif isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.BatchNorm2d, layers.MaskedBatchNorm2d)):
            num_params = layer.weight.numel()
            square_l2_norm = torch.sum(layer.weight**2)
            if layer.bias is not None:
                num_params += layer.bias.numel()
                square_l2_norm += torch.sum(layer.bias**2)
            layer_stats = layers.LayerStats(
                num_params=num_params,
                num_active_params=num_params,
                num_sparse_params=0,
                num_active_sparse_params=0,
                sq_l2_norm=square_l2_norm,
            )
            layer_stats_list.append(layer_stats)

            model_num_active_params += num_params
            model_num_params += num_params

            model_sq_l2_norm += square_l2_norm
        else:
            # ReLU, MaxPool, etc.
            pass

    return layers.ModelStats(
        layer_stats=layer_stats_list,
        num_params=model_num_params,
        num_active_params=model_num_active_params,
        num_sparse_params=model_num_sparse_params,
        num_active_sparse_params=model_num_active_sparse_params,
        sq_l2_norm=model_sq_l2_norm,
    )
