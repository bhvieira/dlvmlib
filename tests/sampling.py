"""Monte Carlo averages of objectives and their gradients, with standard errors."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import torch
from torch import Tensor, nn

from dlvmlib import VAE

Objective = Callable[[VAE, Tensor], Tensor]


def encoder_parameters(vae: VAE) -> list[nn.Parameter]:
    return list(vae.encoder.parameters())


def model_parameters(vae: VAE) -> list[nn.Parameter]:
    return list(vae.decoder.parameters()) + list(vae.prior.parameters())


def flat(tensors: Sequence[Tensor]) -> Tensor:
    return torch.cat([t.flatten() for t in tensors])


def gradient(objective: Objective, vae: VAE, x: Tensor, params: list[nn.Parameter]) -> Tensor:
    """Returns the gradient of `objective(vae, x).mean()` in `params`, flattened."""
    return flat(torch.autograd.grad(objective(vae, x).mean(), params))


def mean_gradient(
    objective: Objective, vae: VAE, x: Tensor, params: list[nn.Parameter], reps: int, batches: int
) -> tuple[Tensor, Tensor]:
    """Returns the mean and the standard error of the gradient at one datapoint `x`.

    Each batch replicates `x` `reps` times, so its gradient averages `reps` independent
    estimates; the standard error comes from the spread across `batches` batches.
    """
    means = torch.stack([gradient(objective, vae, x.expand(reps, -1), params) for _ in range(batches)])
    return means.mean(0), means.std(0) / math.sqrt(batches)


def mean_value(objective: Objective, vae: VAE, x: Tensor, reps: int, batches: int) -> tuple[Tensor, Tensor]:
    """Returns the mean and the standard error of the objective's value at one datapoint `x`."""
    with torch.no_grad():
        means = torch.stack([objective(vae, x.expand(reps, -1)).mean() for _ in range(batches)])
    return means.mean(), means.std() / math.sqrt(batches)
