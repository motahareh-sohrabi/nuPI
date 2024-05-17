import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from .concrete_utils import concrete_cdf, concrete_quantile, sample_eps_noise
from .layer_stats import LayerStats

LayerMasks = tuple[Tensor, Optional[Tensor]]
MaskedForwardOutput = tuple[Tensor, Optional[Tensor]]


class BaseL0Layer(nn.Module):
    """
    Class for sparsifiable "L0" layers with hard-concrete stochastic gates.
    Based on work by C. Louizos, M. Welling, and D. P. Kingma. "Learning Sparse
    Neural Networks through L0 Regularization". In ICLR, 2018.

    Major code re-use from repos:
    - https://github.com/AMLab-Amsterdam/L0_regularization
    - https://github.com/gallego-posada/constrained_sparsity
    """

    def __init__(
        self, l2_detach_gates: bool = True, droprate_init: float = 0.5, temperature: float = 2.0 / 3.0, *args, **kwargs
    ):
        if droprate_init <= 0.0 or droprate_init >= 1.0:
            ValueError("expected droprate_init in (0,1). Got {}".format(droprate_init))
        if temperature <= 0.0 or temperature >= 1.0:
            ValueError("expected temperature in (0,1). Got {}".format(temperature))

        super(BaseL0Layer, self).__init__(*args, **kwargs)

        self.l2_detach_gates = l2_detach_gates
        self.temperature = temperature
        self.droprate_init = droprate_init

    def init_gates_parameters(self):
        """Initialize layer parameters, including the {weight, bias}_log_alpha parameters."""

        # Initialize gate parameters
        gate_mean_init = math.log((1 - self.droprate_init) / self.droprate_init)
        self.weight_log_alpha.data.normal_(gate_mean_init, 1e-2)

        # Weights are scaled by their gates when computing forwards (see get_params()).
        # Thus, their effective intialization is shrunk by a factor of 1 - droprate_init.
        # In order to keep the initialization of L0 and non-L0 layers consistent,
        # we counter this shrinkage by adjusting the initial weights by the same amount.
        initial_sigmoid = 1 - self.droprate_init
        self.weight.data = self.weight.data / initial_sigmoid

    def sample_gates(self, is_test_time: bool = False) -> Tensor:
        """Obtain samples for the stochastic gates. Active gates may have
        fractional values (not necessarily binary 0/1).

        Args:
            is_test_time: If `True`, use the test-time version of the gates.
        """

        factory_kwargs = {"device": self.weight.device, "dtype": self.weight.dtype}

        if not is_test_time:
            # Sample fractional gates in [0,1] based on sampled epsilon
            weight_noise = sample_eps_noise(self.gates_shape, **factory_kwargs)
        else:
            # At test time, choose median of the distribution: 50% quantile
            weight_noise = 0.5 * torch.ones(self.gates_shape, **factory_kwargs)

        weight_z = concrete_quantile(weight_noise, self.weight_log_alpha, self.temperature)
        weight_z = torch.clamp(weight_z, min=0, max=1)

        bias_z = None

        return weight_z, bias_z

    def get_parameters(self, weight_z: Tensor, bias_z: Optional[Tensor] = None) -> tuple[Tensor, Optional[Tensor]]:
        if weight_z is None:
            masked_weight = self.weight
        else:
            # Broadcast weight_z to the shape of the weight tensor
            if not weight_z.shape == self.weight.shape:
                if isinstance(self, nn.Conv2d):
                    weight_z = weight_z.unsqueeze(1).unsqueeze(2).unsqueeze(3)
                elif isinstance(self, nn.Linear):
                    weight_z = weight_z.unsqueeze(0)

            masked_weight = self.weight * weight_z

        if self.bias is not None and bias_z is not None:
            masked_bias = self.bias * bias_z
        else:
            masked_bias = self.bias

        return masked_weight, masked_bias

    def active_gate_probability(self) -> Tensor:
        """Computes the probability that each of the gates is active (i.e. not turned
        off). The returned tensor has the same shape as the gates.
        """

        # *In*active probability per gate
        gate_q0 = concrete_cdf(x=0, log_alpha=self.weight_log_alpha, temperature=self.temperature)

        # P[z > 0] = 1 - P[z =< 0] = 1 - CDF_z(0)
        return 1 - gate_q0

    def expected_l0_norm(self, active_probs: torch.Tensor) -> torch.Tensor:
        # For unstuctured layers, the L0 norm is the sum of the active probabilities
        # since we have one gate per weight. For structured layers, the L0 norm is
        return torch.sum(active_probs) * self.group_size

    def expected_sq_l2_norm(self, active_probs: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def eval_l0_norm(self) -> torch.Tensor:
        weight_z, _ = self.sample_gates(is_test_time=True)
        weight_l0_norm = torch.sum(weight_z * self.group_size)

        bias_l0_norm = 0
        if self.bias is not None and self.is_structured_conv:
            bias_l0_norm = torch.sum(torch.count_nonzero(self.bias))

        return bias_l0_norm + weight_l0_norm

    def eval_sq_l2_norm(self) -> torch.Tensor:
        weight_z, _ = self.sample_gates(is_test_time=True)
        weight, bias = self.get_parameters(weight_z, None)

        weight_sq_l2_norm = torch.linalg.vector_norm(weight, ord=2) ** 2
        bias_sq_l2_norm = 0.0
        if self.bias is not None:
            bias_sq_l2_norm = torch.linalg.vector_norm(bias, ord=2) ** 2

        return weight_sq_l2_norm + bias_sq_l2_norm

    def layer_stats(self, is_test_time: bool) -> LayerStats:

        if not is_test_time:
            # Training time: compute expected L0 an L2 norms
            active_probs = self.active_gate_probability()
            num_active_sparse_params = self.expected_l0_norm(active_probs=active_probs)
            sq_l2_norm = self.expected_sq_l2_norm(active_probs=active_probs)
        else:
            # Test time: use the test-time version of the gates (the median of their
            # distribution, which is deterministic given the current parameters).
            num_active_sparse_params = self.eval_l0_norm()
            sq_l2_norm = self.eval_sq_l2_norm()

        # Each of the weight entries is sparsifiable in the unstructured case.
        num_sparse_params = self.weight.numel()

        if self.bias is not None and self.is_structured_conv:
            # For convolutional layers with structured sparsity, biases are also
            # sparsifiable by being tied to an output feature map.
            num_sparse_params += self.bias.numel()

        # The expected L0 norm counts the expected number of _sparsifiable_
        # parameters in the layer.
        # Using clone() to keep modifications to num_active_sparse_params from
        # changing the original num_active_params tensor.
        num_active_params = num_active_sparse_params.clone()

        if not self.is_structured_conv:
            # Biases are not sparsifiable for linear layers or unstructured conv layers,
            # but they still count towards the total number of active parameters in the
            # layer. For structured conv layers, biases are sparsifiable and are already
            # accounted for.

            # NOTE: this is *not* num_sparse_params, this is num_active_params.
            num_active_params += 0 if self.bias is None else self.bias.numel()

        num_params = self.weight.numel()
        num_params += 0 if self.bias is None else self.bias.numel()

        num_gate_params = self.weight_log_alpha.numel()

        return LayerStats(
            num_params=num_params,
            num_active_params=num_active_params,
            num_sparse_params=num_sparse_params,
            num_active_sparse_params=num_active_sparse_params,
            sq_l2_norm=sq_l2_norm,
            num_gate_params=num_gate_params,
        )

    def gates_parameters(self):
        yield self.weight_log_alpha

    def clamp_parameters(self):
        """Clamp weight_log_alpha parameters for numerical stability."""
        self.weight_log_alpha.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
