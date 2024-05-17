from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from .layer_stats import LayerStats


class MaskedBatchNorm2d(torch.nn.BatchNorm2d):
    """Implements a batch normalization layer able to handle sparse inputs."""

    def __init__(self, feature_mask: torch.ByteTensor = None, *args, **kwargs):
        super(MaskedBatchNorm2d, self).__init__(*args, **kwargs)

        # When initializing with None, the mask buffer would be ignored when
        # creating the state dict for this module. Methods `get_extra_state` and
        # `set_extra_state` are used to handle this correctly.
        self.register_buffer("_feature_mask", feature_mask, persistent=False)

    def get_extra_state(self):
        return {"_feature_mask": self._feature_mask}

    def set_extra_state(self, state):
        self._feature_mask = state["_feature_mask"]

    def get_parameters(self, masked: bool = True) -> tuple[Tensor, Optional[Tensor]]:
        if self._feature_mask is None or not masked:
            return self.weight, self.bias
        else:
            masked_weight = self.weight[self._feature_mask]

            if self.bias is None:
                masked_bias = self.bias
            else:
                masked_bias = self.bias[self._feature_mask, ...]

            return masked_weight, masked_bias

    @torch.no_grad()
    def update_mask_(self, feature_mask: Tensor):
        self._feature_mask = feature_mask

    def layer_stats(self) -> LayerStats:
        num_params = self.weight.numel()
        num_params += 0 if self.bias is None else self.bias.numel()

        with torch.no_grad():
            # No need to computation of sparsity stats of non-sparsifiable
            # parameters in gradient computational graph.
            _weight, _bias = self.get_parameters(masked=True)

        # `get_parameters` applied a filter, unlike other MaskedLayers which
        # return a sparse tensor. Therefore, here we count the number of
        # elements which remain after the filter is applied.
        num_active_params = _weight.numel()
        num_active_params += 0 if _bias is None else _bias.numel()

        # "Sparse" and "total" metrics match for BN layers.
        num_sparse_params = num_params
        num_active_sparse_params = num_active_params

        # Squared L2 norm of BN layers is not counted towards the total model L2 norm
        sq_l2_norm = 0.0

        return LayerStats(
            layer_type="MaskedBatchNorm",
            num_params=num_params,
            num_active_params=num_active_params,
            num_sparse_params=num_sparse_params,
            num_active_sparse_params=num_active_sparse_params,
            sq_l2_norm=sq_l2_norm,
            num_gate_params=None,
        )

    def forward(self, input: torch.Tensor, feature_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Computes the forward pass of the batch normalization layer and
        correctly handles sparse inputs to avoid a biased update of the
        BatchNorm statistics when a given input feature is masked out.
        """

        if feature_mask is None:
            # If no mask is provided, create a dummy mask with all features active.
            feature_mask = torch.ones(self.num_features, device=input.device, dtype=bool)

        # Update self._feature_mask for future calls to self.get_parameters and
        # self.layer_stats
        self.update_mask_(feature_mask)

        if all(feature_mask == 0):
            # If masked, but the mask would deactivate all the features,
            # just return zeros.
            return torch.zeros_like(input, device=input.device)

        # --------------------------- begin<section> ---------------------------
        # This section is copied and unmodified from torch.nn._BatchNorm.forward
        # ----------------------------------------------------------------------

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO(Pytorch): if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        # ---------------------------- end<section> ----------------------------

        # --------------------------- begin<section> ---------------------------
        # Code below was modified by to use the gate_mask
        # ----------------------------------------------------------------------

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if not self.training or self.track_running_stats:
            _running_mean = self.running_mean[feature_mask]
            _running_var = self.running_var[feature_mask]
        else:
            # If buffers are not to be tracked, ensure that they won't be updated
            _running_mean = None
            _running_var = None

        # Keep all the batch dimension, but filter out all the input channels that were turned off
        _input = input[:, feature_mask, ...]
        _weight = self.weight[feature_mask, ...]
        _bias = self.bias[feature_mask]

        # Do BN on outputs from active units
        masked_output = F.batch_norm(
            _input,
            _running_mean,
            _running_var,
            _weight,
            _bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

        # _running_mean and _running_var were updated internally by F.batch_norm
        if self.training and self.track_running_stats:
            self.running_mean[feature_mask] = _running_mean
            self.running_var[feature_mask] = _running_var

        # Construct final output tensor
        # For inactive units: Apply identity mapping (no BN applied) to return
        # zeros in the forward, but retaining the computational graph for the
        # gradients of the layers before the current BN.
        output = input
        # For active units: Actually return result of BN on the active features.
        output[:, feature_mask, ...] = masked_output

        # ---------------------------- end<section> ----------------------------

        return output
