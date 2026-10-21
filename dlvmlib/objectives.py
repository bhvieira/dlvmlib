"""Objectives that train a `VAE` from `k` importance-weighted draws.

Every objective draws `z_1..z_k ~ q(z | x, c)` and forms the log weights
`log w_i = log p(x, z_i | c) - log q(z_i | x, c)`. It returns a tensor of shape `batch` whose
value is the objective's estimate and whose gradient is the objective's gradient estimator, so
`(-objective(vae, x, c).mean()).backward()` trains the encoder, the decoder and the prior with
one optimizer.

ELBO and IWAE are lower bounds on `log p(x | c)` that the encoder and the model both climb. They
share one computation and differ in how the `k` log weights are reduced. Their `gradient`
argument chooses the encoder's gradient estimator:

                     gradient="reparam"     gradient="stl"     gradient="dreg"
    mean over k      ELBO                   ELBO-STL
    log-mean-exp     IWAE                   IWAE-STL           IWAE-DReG

`"stl"` evaluates `log q` with the encoder's parameters frozen, which leaves the encoder's
gradient only the path through `z`, without the score-function term (Roeder et al., 2017). That
keeps ELBO-STL unbiased, but IWAE-STL is biased for `k > 1`, because IWAE's score-function term
doesn't vanish in expectation (Tucker et al., 2019, who name it STL). `"dreg"` also scales the
gradient reaching each draw by its normalized weight, which removes that bias.

EUBO and CUBO are upper bounds, and SUMO (in `dlvmlib.sumo`) is an unbiased estimate. For these
three the encoder and the model follow different gradients. The encoder descends the upper
bounds, which tightens them without touching the evidence, since `log p(x | c)` doesn't depend
on the encoder. The decoder and the prior can't descend them, because that would lower the
evidence too, so they climb IWAE on the same draws. `two_sided` routes the two gradients.
"""

from __future__ import annotations

import abc
import math
from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor
from torch.distributions import kl_divergence

from dlvmlib.model import VAE


def log_mean_exp(t: Tensor, dim: int = 0) -> Tensor:
    """Returns `log mean exp(t)` along `dim`, computed stably."""
    return torch.logsumexp(t, dim=dim) - math.log(t.shape[dim])


def self_normalized(log_w: Tensor) -> Tensor:
    """Returns `sum_i w_hat_i log w_i` along dim 0, with `w_hat_i = w_i / sum_j w_j`."""
    return (torch.softmax(log_w, dim=0) * log_w).sum(dim=0)


def _valued(ascend: Tensor, value: Tensor) -> Tensor:
    """Returns a tensor with the value of `value` and the gradient of `ascend`."""
    return ascend + (value - ascend).detach()


def _require_draws(k: int, least: int) -> int:
    if k < least:
        raise ValueError(f"k must be at least {least}, got {k}")
    return k


def _require_gradient(gradient: str, *allowed: str) -> str:
    if gradient not in allowed:
        names = ", ".join(map(repr, allowed[:-1])) + f" or {allowed[-1]!r}"
        raise ValueError(f"gradient must be {names}, got {gradient!r}")
    return gradient


class Bound(abc.ABC):
    """A lower bound from `k` draws, the computation that ELBO and IWAE share.

    A subclass chooses the reduction over draws in `reduce`, whether `log q` is evaluated with
    the encoder frozen in `frozen`, and how many groups of `k` draws to take in `repeats`.
    """

    frozen: bool = False
    repeats: int = 1

    def __init__(self, k: int = 1) -> None:
        self.k = _require_draws(k, 1)

    def __call__(self, vae: VAE, x: Tensor, c: Tensor | None = None) -> Tensor:
        z, log_q = vae.draw(x, c, self.k * self.repeats, frozen=self.frozen)
        return self.reduce(vae.log_joint(x, z, c) - log_q, z)

    @abc.abstractmethod
    def reduce(self, log_w: Tensor, z: Tensor) -> Tensor:
        """Reduces the log weights of shape `(k * repeats, *batch)` to the bound, of shape `batch`."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(k={self.k})"


class ELBO(Bound):
    """The evidence lower bound `E_q[log w]`, averaged over `k` draws (Kingma & Welling, 2014).

    With `gradient="stl"`, this is ELBO-STL, the "sticking the landing" encoder gradient
    (Roeder et al., 2017). `log q` is evaluated with the encoder frozen, so the encoder's
    gradient keeps the path through `z` and drops the score-function term, whose expectation is
    zero. The estimate stays unbiased for the ELBO's gradient at any `k`, and its variance
    vanishes as `q` approaches the posterior. The value and the model's gradient don't change.

    With `analytic_kl=True`, the KL divergence from the posterior to the prior is computed in
    closed form, and only the reconstruction term `log p(x | z, c)` is averaged over the draws.
    That needs a closed form registered with `torch.distributions.kl_divergence`, such as the
    one between the `DiagGaussian` encoder and a diagonal Gaussian prior. It leaves no `log q`
    to freeze, so it takes only `gradient="reparam"`.
    """

    def __init__(self, k: int = 1, analytic_kl: bool = False, gradient: Literal["reparam", "stl"] = "reparam") -> None:
        super().__init__(k)
        self.gradient = _require_gradient(gradient, "reparam", "stl")
        if analytic_kl and gradient != "reparam":
            raise ValueError("analytic_kl=True leaves no log q to freeze; use gradient='reparam'")
        self.analytic_kl = analytic_kl
        self.frozen = gradient == "stl"

    def __call__(self, vae: VAE, x: Tensor, c: Tensor | None = None) -> Tensor:
        if not self.analytic_kl:
            return super().__call__(vae, x, c)
        q, p = vae.posterior(x, c), vae.latent_prior(c)
        try:
            kl = kl_divergence(q, p)
        except NotImplementedError:
            raise TypeError(
                f"no closed-form KL divergence from {type(q).__name__} to {type(p).__name__}; use analytic_kl=False"
            ) from None
        z = q.rsample((self.k,))
        log_px = vae.likelihood(z, c).log_prob(x.expand(*z.shape[:-1], x.shape[-1]))
        return log_px.mean(dim=0) - kl

    def reduce(self, log_w: Tensor, z: Tensor) -> Tensor:
        return log_w.mean(dim=0)

    def __repr__(self) -> str:
        return f"ELBO(k={self.k}, analytic_kl={self.analytic_kl}, gradient={self.gradient!r})"


class IWAE(Bound):
    """The importance-weighted bound `log mean_i w_i` (Burda et al., 2016).

    It's at least the ELBO on the same draws, and it approaches `log p(x | c)` as `k` grows.
    As `k` grows, the encoder's reparameterized gradient loses its signal relative to its noise
    (Rainforth et al., 2018); IWAE-DReG avoids that.

    With `gradient="stl"`, this is IWAE-STL (Roeder et al., 2017): `log q` is evaluated with the
    encoder frozen, which drops the score-function term. For IWAE that term doesn't vanish in
    expectation, so IWAE-STL is biased for `k > 1` (Tucker et al., 2019, who name it STL).

    With `gradient="dreg"`, this is IWAE-DReG, the doubly reparameterized encoder gradient
    (Tucker et al., 2019). `log q` is evaluated with the encoder frozen, and the gradient that
    reaches each draw `z_i` is scaled once more by its normalized weight
    `w_hat_i = w_i / sum_j w_j`. The encoder then receives
    `sum_i w_hat_i^2 (d log w_i / d z_i) (d z_i / d phi)`, an unbiased estimate of IWAE's
    gradient without the score-function term.

    Whatever the gradient, the value and the model's gradient are IWAE's.

    With `repeats=m`, this is MIWAE (Rainforth et al., 2018), the mean of `m` independent IWAE
    bounds of `k` draws each. It takes `k * m` draws and reduces each group of `k` on its own, so
    with `gradient="dreg"` the weights `w_hat_i` are normalized within each group. Its
    expectation is that of `IWAE(k)`, looser than `IWAE(k * m)` on as many draws. In exchange,
    the reparameterized encoder gradient's signal-to-noise ratio grows with `m`, where it falls
    with `k` (Rainforth et al., 2018).
    """

    def __init__(self, k: int, gradient: Literal["reparam", "stl", "dreg"] = "reparam", repeats: int = 1) -> None:
        super().__init__(k)
        self.gradient = _require_gradient(gradient, "reparam", "stl", "dreg")
        if repeats < 1:
            raise ValueError(f"repeats must be at least 1, got {repeats}")
        self.repeats = repeats
        self.frozen = gradient != "reparam"

    def reduce(self, log_w: Tensor, z: Tensor) -> Tensor:
        log_w = log_w.unflatten(0, (self.k, self.repeats))
        if self.gradient == "dreg" and z.requires_grad:
            weights = torch.softmax(log_w.detach(), dim=0).flatten(0, 1).unsqueeze(-1)
            z.register_hook(lambda grad: grad * weights)
        return log_mean_exp(log_w).mean(dim=0)

    def __repr__(self) -> str:
        return f"IWAE(k={self.k}, gradient={self.gradient!r}, repeats={self.repeats})"


def two_sided(
    vae: VAE,
    x: Tensor,
    c: Tensor | None,
    k: int,
    *,
    model: Callable[[Tensor], Tensor],
    encoder: Callable[[Tensor], Tensor],
    value: Callable[[Tensor], Tensor],
) -> Tensor:
    """Returns a tensor whose value is `value(log w)` and whose gradient trains each side apart.

    Each argument maps log weights of shape `(k, *batch)` to shape `batch`. The decoder and the
    prior climb `model(log w)`, computed from detached draws so that nothing reaches the
    encoder. The encoder descends `encoder(log w)`, computed with the decoder and the prior
    frozen so that nothing reaches them. The decoder and the prior are therefore evaluated
    twice.
    """
    z, log_q = vae.draw(x, c, k)
    model_log_w = vae.log_joint(x, z.detach(), c) - log_q.detach()
    encoder_log_w = vae.log_joint(x, z, c, frozen=True) - log_q
    return _valued(model(model_log_w) - encoder(encoder_log_w), value(encoder_log_w.detach()))


class EUBO:
    """The evidence upper bound `E_p(z|x,c)[log w] = log p(x | c) + KL(p(z | x, c) || q)` (Ji & Shen, 2019).

    The value is the self-normalized estimate `sum_i w_hat_i log w_i` from their Eq. 6, with
    `w_hat_i = w_i / sum_j w_j`. The encoder descends it, which shrinks `KL(p || q)`.

    With `gradient="reparam"`, the encoder's gradient is the reparameterized derivative of that
    estimate, weights included. With `gradient="score"`, it's their score form,
    `-sum_i w_hat_i d log q(z_i) / d phi` at fixed draws, which also trains a posterior that
    can't be reparameterized. Their Eq. 5 isn't offered. It leaves the factor `log w + 1` off
    the `log q` term, so it isn't the gradient of the bound and it changes when the joint is
    scaled by a constant.

    The decoder and the prior climb IWAE on the same draws. With the weights held fixed, that's
    the gradient of the EUBO estimate itself, `sum_i w_hat_i d log p(x, z_i) / d theta`, which
    estimates `d log p(x | c) / d theta` by Fisher's identity. Differentiating through the
    weights would add `d KL(p || q) / d theta`, so the model could raise the bound by moving its
    posterior away from `q` rather than by raising the evidence.

    The estimate is biased at finite `k` and can fall below `log p(x | c)`. It needs `k >= 2`.
    """

    def __init__(self, k: int, gradient: Literal["reparam", "score"] = "reparam") -> None:
        self.gradient = _require_gradient(gradient, "reparam", "score")
        self.k = _require_draws(k, 2)

    def __call__(self, vae: VAE, x: Tensor, c: Tensor | None = None) -> Tensor:
        if self.gradient == "reparam":
            return two_sided(vae, x, c, self.k, model=log_mean_exp, encoder=self_normalized, value=self_normalized)
        z, log_q = vae.draw(x, c, self.k, reparameterized=False)
        log_w = vae.log_joint(x, z, c) - log_q.detach()
        weights = torch.softmax(log_w.detach(), dim=0)
        return _valued(log_mean_exp(log_w) + (weights * log_q).sum(dim=0), self_normalized(log_w.detach()))

    def __repr__(self) -> str:
        return f"EUBO(k={self.k}, gradient={self.gradient!r})"


class CUBO:
    """The chi upper bound `(1 / n) log E_q[w^n]`, for `n > 1` (Dieng et al., 2017).

    The value is the `k`-draw estimate `(1 / n) log mean_i w_i^n`. It's biased low, so it isn't
    guaranteed to stay above `log p(x | c)` when `k` is small (their Section 2.3).

    The encoder descends `exp(n CUBO)` instead, which has the same minimizer and whose `k`-draw
    estimate is unbiased. Its reparameterized gradient is `n mean_i w_i^n d log w_i / d phi`
    (their Eq. 3). Each datapoint's gradient is divided by `n exp(n max_i log w_i)`. Their
    Algorithm 1, a score-function version, divides by `exp(n max_i log w_i)` to avoid underflow,
    and the division prevents overflow too. With a single draw that rescaling leaves no gradient,
    so CUBO needs `k >= 2`.

    The rescaling depends on the draws, so it gives up the unbiasedness of Eq. 3: each step
    points along the gradient of the biased `k`-draw estimate. That estimate falls without
    bound as `q` narrows, because few draws never reach the heavy tail of `w`. With few draws
    the encoder can therefore collapse, and on the linear-Gaussian test model it does at
    `k = 8` while it converges at `k = 64`. The number of draws needed grows with the latent
    dimension.

    The decoder and the prior climb IWAE on the same draws, as in `EUBO`.
    """

    def __init__(self, k: int, n: float = 2.0) -> None:
        if n <= 1:
            raise ValueError(f"CUBO is an upper bound for n > 1, got n={n}")
        self.k = _require_draws(k, 2)
        self.n = n

    def __call__(self, vae: VAE, x: Tensor, c: Tensor | None = None) -> Tensor:
        return two_sided(vae, x, c, self.k, model=log_mean_exp, encoder=self._exponentiated, value=self._estimate)

    def _estimate(self, log_w: Tensor) -> Tensor:
        return log_mean_exp(self.n * log_w) / self.n

    def _exponentiated(self, log_w: Tensor) -> Tensor:
        scaled = self.n * (log_w - log_w.detach().amax(dim=0))
        return scaled.exp().mean(dim=0) / self.n

    def __repr__(self) -> str:
        return f"CUBO(k={self.k}, n={self.n})"
