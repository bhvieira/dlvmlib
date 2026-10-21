"""SUMO: an unbiased estimate of `log p(x | c)` by Russian roulette over IWAE (Luo et al., 2020).

`IWAE_j`, the IWAE bound with `j` draws, increases in expectation toward `log p(x | c)` as `j`
grows. SUMO writes the limit as the telescoping series
`IWAE_m + sum_{j >= 1} (IWAE_{m+j} - IWAE_{m+j-1})`, draws a random number of terms `K`, and
divides the `j`-th difference by `P(K >= j)`:

    SUMO = IWAE_m + sum_{j=1}^{K} (IWAE_{m+j} - IWAE_{m+j-1}) / P(K >= j)

For any tail with `P(K >= j) > 0`, its expectation is `log p(x | c)`. One evaluation costs
`m + K` draws, and every `IWAE_j` comes from the first `j` of them.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from dlvmlib.model import VAE
from dlvmlib.objectives import two_sided


class RouletteTail:
    """The distribution of the number of roulette terms `K`, given by its tail `P(K >= j)`.

    `P(K >= j)` is `1 / j` for `j < alpha` and `(1 / alpha) (1 - decay)^(j - alpha)` from
    `alpha` on (Luo et al., 2020, Eq. 10). The `1 / j` part balances the estimator's variance
    against its cost. The geometric part keeps the expected number of terms finite: about 5 at
    the default `alpha = 80` and `decay = 0.1`.
    """

    def __init__(self, alpha: int = 80, decay: float = 0.1) -> None:
        if alpha < 1:
            raise ValueError(f"alpha must be at least 1, got {alpha}")
        if not 0 < decay < 1:
            raise ValueError(f"decay must lie in (0, 1), got {decay}")
        self.alpha = alpha
        self.decay = decay

    def survival(self, j: Tensor) -> Tensor:
        """Returns `P(K >= j)` for integers `j >= 1`."""
        j = j.to(torch.get_default_dtype())
        geometric = (1 - self.decay) ** (j - self.alpha) / self.alpha
        return torch.where(j < self.alpha, 1 / j, geometric)

    def terms(self, u: Tensor) -> Tensor:
        """Returns `K` for uniform draws `u` in `(0, 1]`, by inverting the tail.

        `K = max{j : P(K >= j) >= u}`, so `P(K >= j) = P(u <= P(K >= j))` as required.
        """
        harmonic = torch.floor(1 / u)
        geometric = self.alpha + torch.floor(torch.log(self.alpha * u) / math.log(1 - self.decay))
        return torch.where(u > 1 / self.alpha, harmonic, geometric).long()

    def sample(self) -> int:
        """Draws one `K`."""
        return int(self.terms(1 - torch.rand(())).item())

    def mean(self) -> float:
        """Returns the expected number of terms, `sum_{j >= 1} P(K >= j)`."""
        return sum(1 / j for j in range(1, self.alpha)) + 1 / (self.alpha * self.decay)

    def __repr__(self) -> str:
        return f"RouletteTail(alpha={self.alpha}, decay={self.decay})"


def sumo_estimate(log_w: Tensor, m: int, inverse_tail: Tensor) -> Tensor:
    """Returns SUMO from `m + K` log weights along dim 0 and the `K` weights `1 / P(K >= j)`.

    The result has the shape of `log_w` without its first axis.

    Raises:
        ValueError: If `m` isn't between 1 and the number of draws, or if `inverse_tail` doesn't
            hold one weight for each of the remaining draws.
    """
    draws = log_w.shape[0]
    if not 1 <= m <= draws:
        raise ValueError(f"m must lie between 1 and the {draws} draws, got {m}")
    if inverse_tail.shape != (draws - m,):
        raise ValueError(
            f"{draws} draws with m={m} need {draws - m} roulette weights, got shape {tuple(inverse_tail.shape)}"
        )
    counts = torch.arange(1, draws + 1, dtype=log_w.dtype, device=log_w.device)
    column = (-1,) + (1,) * (log_w.dim() - 1)
    iwae = torch.logcumsumexp(log_w, dim=0) - counts.log().view(column)
    differences = iwae[m:] - iwae[m - 1 : -1]
    return iwae[m - 1] + (inverse_tail.to(log_w).view(column) * differences).sum(dim=0)


class SUMO:
    """SUMO, trained as in Luo et al. (2020).

    The value is SUMO, with `IWAE_m` computed in full and the rest estimated by roulette. The
    decoder and the prior climb SUMO, whose gradient is an unbiased estimate of
    `d log p(x | c) / d theta`. SUMO's expectation doesn't depend on the encoder, so the encoder
    descends `SUMO^2` instead, whose gradient is an unbiased estimate of the gradient of SUMO's
    variance (their Eq. 12). That gradient is occasionally very large, and the paper clips the
    encoder's gradients (their Section 3.3), and in its density-modeling experiments the
    decoder's too (their Appendix A.8).

    One `K` is drawn per call, as in their Algorithm 1. Algorithm 1 handles a single datapoint,
    so sharing `K` across the batch is this library's choice. Every datapoint's estimate stays
    unbiased.
    """

    def __init__(self, m: int = 1, tail: RouletteTail | None = None) -> None:
        if m < 1:
            raise ValueError(f"m must be at least 1, got {m}")
        self.m = m
        self.tail = tail if tail is not None else RouletteTail()

    def __call__(self, vae: VAE, x: Tensor, c: Tensor | None = None) -> Tensor:
        terms = self.tail.sample()
        inverse_tail = 1 / self.tail.survival(torch.arange(1, terms + 1))

        def estimate(log_w: Tensor) -> Tensor:
            return sumo_estimate(log_w, self.m, inverse_tail)

        def square(log_w: Tensor) -> Tensor:
            return estimate(log_w).square()

        return two_sided(vae, x, c, self.m + terms, model=estimate, encoder=square, value=estimate)

    def __repr__(self) -> str:
        return f"SUMO(m={self.m}, tail={self.tail!r})"
