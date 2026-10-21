"""The model: a prior, a decoder and an amortized posterior, each a zuko lazy distribution.

A `VAE` is the latent variable model `p(x, z | c) = p(x | z, c) p(z | c)` together with an
encoder `q(z | x, c)` that proposes latents for importance sampling. `c` holds optional
covariates. The encoder reads `cat(c, x)`, the decoder reads `cat(c, z)`, and the prior reads
`c` only when it is conditional. Every part is a `zuko.lazy.LazyDistribution`: a Gaussian MLP,
a normalizing flow or a mixture can be the encoder or the decoder, and a flow, a mixture or a
fixed distribution can be the prior. A Gaussian MLP needs a context, so it can be the prior only
when the prior is conditional. A mixture can't be reparameterized, so as the encoder it trains
only by `EUBO(gradient="score")`.

Draws come first in every shape. `k` draws for data of shape `(*batch, features)` give
latents of shape `(k, *batch, latent)` and log densities of shape `(k, *batch)`.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from zuko.distributions import NormalizingFlow
from zuko.lazy import LazyDistribution


def _context(c: Tensor | None, t: Tensor) -> Tensor:
    """Returns `cat(c, t)` along the last axis, with `c` broadcast to the leading shape of `t`."""
    if c is None:
        return t
    return torch.cat((c.expand(*t.shape[:-1], c.shape[-1]), t), dim=-1)


def _live(module: nn.Module, *args: Tensor) -> Distribution:
    return module(*args)


def _frozen(module: nn.Module, *args: Tensor) -> Distribution:
    """Calls `module` with its parameters detached, so gradients reach only the arguments."""
    detached = {name: p.detach() for name, p in module.named_parameters()}
    return torch.func.functional_call(module, detached, args)


class VAE(nn.Module):
    """A deep latent variable model with an amortized posterior.

    `encoder` is `q(z | x, c)`, `decoder` is `p(x | z, c)`, and `prior` is `p(z)`, or
    `p(z | c)` when `conditional_prior` is set. Without covariates, the encoder reads `x` and
    the decoder reads `z`. With covariates of shape `(*batch, covariates)`, the encoder's
    context has `covariates + features` columns and the decoder's has `covariates + latent`.

    The objectives in `dlvmlib.objectives` and `dlvmlib.sumo` train it.
    """

    def __init__(
        self,
        encoder: LazyDistribution,
        decoder: LazyDistribution,
        prior: LazyDistribution,
        conditional_prior: bool = False,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior
        self.conditional_prior = conditional_prior

    def posterior(self, x: Tensor, c: Tensor | None = None) -> Distribution:
        """Returns the encoder's `q(z | x, c)`, with batch shape `x.shape[:-1]`."""
        return self.encoder(_context(c, x))

    def likelihood(self, z: Tensor, c: Tensor | None = None) -> Distribution:
        """Returns the decoder's `p(x | z, c)`, with batch shape `z.shape[:-1]`."""
        return self.decoder(_context(c, z))

    def latent_prior(self, c: Tensor | None = None) -> Distribution:
        """Returns the prior `p(z | c)`, or `p(z)` when the prior isn't conditional."""
        return self.prior(*self._prior_args(c))

    def draw(
        self,
        x: Tensor,
        c: Tensor | None = None,
        k: int = 1,
        *,
        frozen: bool = False,
        reparameterized: bool = True,
    ) -> tuple[Tensor, Tensor]:
        """Returns `k` draws `z ~ q(z | x, c)` and their log densities `log q(z | x, c)`.

        The draws are reparameterized, so gradients reach the encoder through `z`. A zuko
        normalizing flow draws `z` and `log q` in one pass, so a flow that samples cheaply is
        never inverted. With `frozen=True`, `log q` is evaluated with the encoder's parameters
        detached, so its gradient reaches the encoder only through `z`. That removes the
        score-function term, at the cost of one more pass through the encoder. With
        `reparameterized=False`, `z` is drawn without gradient, and `log q` then carries only
        the score-function term; this suits a posterior that can't be reparameterized.

        Raises:
            ValueError: If the posterior can't be reparameterized and `reparameterized` is set.
        """
        context = _context(c, x)
        q = self.encoder(context)
        if not reparameterized:
            z = q.sample((k,))
            return z, q.log_prob(z)
        if not q.has_rsample:
            raise ValueError(
                f"{type(q).__name__} can't be reparameterized; only EUBO(gradient='score') trains such an encoder"
            )
        if frozen:
            z = q.rsample((k,))
            return z, _frozen(self.encoder, context).log_prob(z)
        if isinstance(q, NormalizingFlow):
            return q.rsample_and_log_prob((k,))
        z = q.rsample((k,))
        return z, q.log_prob(z)

    def log_joint(self, x: Tensor, z: Tensor, c: Tensor | None = None, *, frozen: bool = False) -> Tensor:
        """Returns `log p(x | z, c) + log p(z | c)` for latents `z` of shape `(k, *batch, latent)`.

        The result has shape `(k, *batch)`. With `frozen=True`, the decoder and the prior are
        evaluated with their parameters detached, so the gradient reaches only `z`.
        """
        call = _frozen if frozen else _live
        log_px = call(self.decoder, _context(c, z)).log_prob(x.expand(*z.shape[:-1], x.shape[-1]))
        return log_px + call(self.prior, *self._prior_args(c)).log_prob(z)

    @torch.no_grad()
    def log_marginal(self, x: Tensor, c: Tensor | None = None, k: int = 5000, chunk: int = 500) -> Tensor:
        """Estimates `log p(x | c)` by importance sampling with `k` draws from the encoder.

        The estimate is the IWAE bound with `k` draws, the usual held-out log-likelihood
        (Burda et al., 2016). It's a lower bound whose gap shrinks as `k` grows. The draws are
        taken `chunk` at a time to bound memory. The result has shape `x.shape[:-1]`.
        """
        if k < 1 or chunk < 1:
            raise ValueError(f"k and chunk must be at least 1, got k={k} and chunk={chunk}")
        log_sum = None
        for start in range(0, k, chunk):
            z, log_q = self.draw(x, c, min(chunk, k - start), reparameterized=False)
            chunk_sum = torch.logsumexp(self.log_joint(x, z, c) - log_q, dim=0)
            log_sum = chunk_sum if log_sum is None else torch.logaddexp(log_sum, chunk_sum)
        assert log_sum is not None
        return log_sum - math.log(k)

    @torch.no_grad()
    def sample(self, n: int, c: Tensor | None = None) -> Tensor:
        """Draws `n` samples `x ~ p(x | c)` by ancestral sampling.

        Without covariates the result has shape `(n, features)`. With covariates of shape
        `(*batch, covariates)`, it has shape `(n, *batch, features)`: `n` samples for each row
        of `c`.
        """
        batch = () if c is None or self.conditional_prior else tuple(c.shape[:-1])
        z = self.latent_prior(c).sample((n, *batch))
        return self.likelihood(z, c).sample()

    def _prior_args(self, c: Tensor | None) -> tuple[Tensor, ...]:
        if not self.conditional_prior:
            return ()
        if c is None:
            raise ValueError("a conditional prior needs covariates c")
        return (c,)
