from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch


@dataclass
class LayerStats:
    """
    Statistics for a MaskedLayer, including sparsity and L2 regularization.

    Args:
        layer_type: Type of layer, e.g. "Linear", "Conv2d", "BatchNorm2d".
        num_params: Total number of trainable parameters in layer. This counts
            both the weights and the biases.
        num_active_params: Number of active parameters in layer for the current
            masks.
        num_sparse_params: Number of sparsifiable parameters in layer. This only
            counts parameters that can be removed through sparsification: for
            example, by applying magnitude pruning, or via trainable gates.
        num_active_sparse_params: Number of active sparsifiable parameters in
            the layer. In the case of layers with trainable gates, this measures
            the _expected_ number of active parameters.
        sq_l2_norm: Squared L2 norm of the layer's parameters. This counts the
            squared L2 norm for both weights and biases. In the case of layers
            with trainable gates, this measures the _expected_ squared L2 norm.
            The L2 norm of BatchNorm layers is not included in this computation
            (set to 0.0).
        num_gate_params: Total number of trainable parameters associated with
            the gates of the layer. For non-gated layers this is `None`.
    """

    num_params: int
    num_active_params: Union[int, torch.Tensor]

    num_sparse_params: int
    num_active_sparse_params: Union[int, torch.Tensor]

    sq_l2_norm: Optional[Union[float, torch.Tensor]] = None

    num_gate_params: Optional[int] = None

    def compute_density_stats(
        self,
    ) -> Tuple[Union[float, torch.Tensor], Optional[Union[float, torch.Tensor]]]:
        """
        Compute density statistics for a layer.
        """

        density = self.num_active_params / self.num_params

        if self.num_sparse_params > 0:
            sparse_density = self.num_active_sparse_params / self.num_sparse_params
        else:
            sparse_density = None

        return density, sparse_density
