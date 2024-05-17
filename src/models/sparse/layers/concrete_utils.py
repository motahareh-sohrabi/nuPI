import math

import torch

# LIMIT_L = gamma; LIMIT_R = zeta -- 'stretching' parameters (Sect 4, p7)
LIMIT_L, LIMIT_R, EPS = -0.1, 1.1, 1e-6


@torch.no_grad()
def sample_eps_noise(size: tuple[int], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Sample uniform noise for the concrete distribution."""

    # Inverse CDF sampling from the concrete distribution. Then clamp for gates
    eps_noise = torch.rand(size, dtype=dtype, device=device)
    # Transform to interval (EPS, 1-EPS)
    eps_noise = EPS + (1 - 2 * EPS) * eps_noise

    return eps_noise


def concrete_cdf(x: float, log_alpha: torch.Tensor, temperature: float) -> torch.Tensor:
    """Implements the CDF of the 'stretched' concrete distribution."""
    # 'Stretch' input to (gamma, zeta) -- Eq 25 (appendix).
    x_stretched = (x - LIMIT_L) / (LIMIT_R - LIMIT_L)

    # Eq 24 (appendix)
    pre_sigmoid = math.log(x_stretched / (1 - x_stretched))
    pre_clamp = torch.sigmoid(pre_sigmoid * temperature - log_alpha)

    return pre_clamp.clamp(min=EPS, max=1 - EPS)


def concrete_quantile(x: float, log_alpha: torch.Tensor, temperature: float) -> torch.Tensor:
    """Implements the quantile, aka inverse CDF, of the 'stretched' concrete
    distribution, given a uniform sample x (Eq. 10)."""

    y = torch.sigmoid((torch.log(x / (1 - x)) + log_alpha) / temperature)
    return y * (LIMIT_R - LIMIT_L) + LIMIT_L
