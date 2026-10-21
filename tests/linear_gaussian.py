"""A linear-Gaussian latent variable model, where the evidence and the posterior are closed form.

    z ~ N(0, I),   x | z ~ N(W z + b, s^2 I)
    x ~ N(b, W W^T + s^2 I)
    z | x ~ N(P W^T (x - b) / s^2, P),   P = (I + W^T W / s^2)^-1

`W` has orthogonal columns, so `P` is diagonal and the affine Gaussian encoder
`q(z | x) = N(A x + d, diag t^2)` contains the exact posterior. Against this model the ELBO,
EUBO and CUBO are closed form too, as functions of the encoder's parameters.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, MultivariateNormal
from zuko.distributions import DiagNormal
from zuko.lazy import LazyDistribution, UnconditionalDistribution

from dlvmlib import VAE

FEATURES, LATENT = 3, 2

# Directions in which a mismatched encoder leaves the exact posterior. `log_t` moves up, so the
# encoder is wider than the posterior and every moment the tests need stays finite.
_A_OFFSET = torch.tensor([[0.3, -0.2, 0.1], [0.1, 0.2, -0.3]])
_D_OFFSET = torch.tensor([0.3, -0.2])
_LOG_T_OFFSET = torch.tensor([0.25, 0.15])


class LinearDecoder(LazyDistribution):
    """`p(x | z) = N(W z + b, s^2 I)`."""

    def __init__(self) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.tensor([[1.2, 0.0], [0.0, -0.8], [0.5, 0.0]]))
        self.b = nn.Parameter(torch.tensor([0.3, -0.1, 0.2]))
        self.log_s = nn.Parameter(torch.tensor(math.log(0.7)))

    def forward(self, c: Tensor | None = None) -> Distribution:
        assert c is not None
        loc = c @ self.W.T + self.b
        return DiagNormal(loc, self.log_s.exp().expand_as(loc))


class AffineEncoder(LazyDistribution):
    """`q(z | x) = N(A x + d, diag t^2)`."""

    def __init__(self, A: Tensor, d: Tensor, log_t: Tensor) -> None:
        super().__init__()
        self.A = nn.Parameter(A.clone())
        self.d = nn.Parameter(d.clone())
        self.log_t = nn.Parameter(log_t.clone())

    def forward(self, c: Tensor | None = None) -> Distribution:
        assert c is not None
        loc = c @ self.A.T + self.d
        return DiagNormal(loc, self.log_t.exp().expand_as(loc))


def exact_encoder(decoder: LinearDecoder) -> tuple[Tensor, Tensor, Tensor]:
    """Returns the `(A, d, log_t)` at which the encoder equals the posterior."""
    W, b, s2 = decoder.W, decoder.b, (2 * decoder.log_s).exp()
    P = torch.linalg.inv(torch.eye(LATENT) + W.T @ W / s2)
    A = P @ W.T / s2
    return A, -A @ b, 0.5 * torch.diagonal(P).log()


def build(mismatch: float = 0.0) -> VAE:
    """Returns the model with its encoder `mismatch` units away from the exact posterior."""
    decoder = LinearDecoder()
    with torch.no_grad():
        A, d, log_t = exact_encoder(decoder)
    encoder = AffineEncoder(A + mismatch * _A_OFFSET, d + mismatch * _D_OFFSET, log_t + mismatch * _LOG_T_OFFSET)
    prior = UnconditionalDistribution(DiagNormal, torch.zeros(LATENT), torch.ones(LATENT), buffer=True)
    return VAE(encoder, decoder, prior)


def data(n: int = 4) -> Tensor:
    """Returns `n` fixed datapoints of shape `(n, FEATURES)`."""
    generator = torch.Generator().manual_seed(1234)
    return torch.randn(n, FEATURES, generator=generator)


def _parts(vae: VAE) -> tuple[LinearDecoder, AffineEncoder]:
    decoder, encoder = vae.decoder, vae.encoder
    assert isinstance(decoder, LinearDecoder) and isinstance(encoder, AffineEncoder)
    return decoder, encoder


def log_evidence(vae: VAE, x: Tensor) -> Tensor:
    """Returns `log p(x)`, differentiable in the decoder's parameters."""
    decoder, _ = _parts(vae)
    covariance = decoder.W @ decoder.W.T + (2 * decoder.log_s).exp() * torch.eye(FEATURES)
    return MultivariateNormal(decoder.b, covariance).log_prob(x)


def posterior(vae: VAE, x: Tensor) -> tuple[Tensor, Tensor]:
    """Returns the mean and the variances of `p(z | x)`."""
    decoder, _ = _parts(vae)
    A, d, log_t = exact_encoder(decoder)
    return x @ A.T + d, (2 * log_t).exp().expand(*x.shape[:-1], LATENT)


def _encoder_moments(vae: VAE, x: Tensor) -> tuple[Tensor, Tensor]:
    _, encoder = _parts(vae)
    return x @ encoder.A.T + encoder.d, (2 * encoder.log_t).exp().expand(*x.shape[:-1], LATENT)


def elbo(vae: VAE, x: Tensor) -> Tensor:
    """Returns `E_q[log p(x, z) - log q(z | x)]`, differentiable in every parameter."""
    decoder, _ = _parts(vae)
    mu, t2 = _encoder_moments(vae, x)
    s2 = (2 * decoder.log_s).exp()
    residual = x - mu @ decoder.W.T - decoder.b
    spread = t2 @ (decoder.W**2).T
    normalizer = -0.5 * FEATURES * math.log(2 * math.pi) - FEATURES * decoder.log_s
    reconstruction = normalizer - ((residual**2) + spread).sum(-1) / (2 * s2)
    kl = 0.5 * (t2 + mu**2 - 1 - t2.log()).sum(-1)
    return reconstruction - kl


def eubo(vae: VAE, x: Tensor) -> Tensor:
    """Returns `log p(x) + KL(p(z | x) || q)`, differentiable in the encoder's parameters."""
    m, v = posterior(vae, x)
    mu, t2 = _encoder_moments(vae, x)
    kl = 0.5 * (t2.log() - v.log() + (v + (m - mu) ** 2) / t2 - 1).sum(-1)
    return log_evidence(vae, x).detach() + kl


def cubo(vae: VAE, x: Tensor, n: float) -> Tensor:
    """Returns `(1 / n) log E_q[w^n]`, differentiable in the encoder's parameters.

    Per latent dimension, `E_q[(p / q)^n]` is the Gaussian integral of `p^n q^(1 - n)`.
    """
    m, v = posterior(vae, x)
    mu, t2 = _encoder_moments(vae, x)
    a = n / v + (1 - n) / t2
    beta = n * m / v + (1 - n) * mu / t2
    gamma = n * m**2 / v + (1 - n) * mu**2 / t2
    log_integral = (
        -0.5 * n * torch.log(2 * math.pi * v)
        - 0.5 * (1 - n) * torch.log(2 * math.pi * t2)
        + 0.5 * torch.log(2 * math.pi / a)
        - 0.5 * (gamma - beta**2 / a)
    )
    return log_evidence(vae, x).detach() + log_integral.sum(-1) / n
