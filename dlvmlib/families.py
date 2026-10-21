"""Conditional distributions whose parameters an MLP reads from the context.

Each is a `zuko.lazy.LazyDistribution`, so it can fill the encoder's or the decoder's role in
a `VAE` next to zuko's flows. `DiagGaussian` is the usual VAE encoder, and the one family here
with a closed-form KL divergence to a diagonal Gaussian prior. `StudentT` suits data with heavy
tails, and `Bernoulli` suits binary data; it can't be reparameterized, so it serves as a
decoder.

Every constructor forwards extra keyword arguments to `zuko.nn.MLP`, such as `activation`.
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor
from torch.distributions import Bernoulli as _Bernoulli
from torch.distributions import Distribution, Independent
from torch.distributions import StudentT as _StudentT
from zuko.distributions import DiagNormal
from zuko.lazy import LazyDistribution
from zuko.nn import MLP


def _required(c: Tensor | None, name: str) -> Tensor:
    if c is None:
        raise ValueError(f"{name} is conditional and needs a context")
    return c


class DiagGaussian(LazyDistribution):
    """A diagonal Gaussian `N(mu(c), diag sigma(c)^2)` over `features` variables.

    The MLP maps `context` features to the mean and the log standard deviation.
    """

    def __init__(self, features: int, context: int, hidden_features: Sequence[int] = (64, 64), **kwargs) -> None:
        super().__init__()
        self.hyper = MLP(context, 2 * features, hidden_features, **kwargs)

    def forward(self, c: Tensor | None = None) -> Distribution:
        loc, log_scale = self.hyper(_required(c, "DiagGaussian")).chunk(2, dim=-1)
        return DiagNormal(loc, log_scale.exp())


class StudentT(LazyDistribution):
    """Independent Student's t variables `t_nu(c)(mu(c), sigma(c))` over `features` variables.

    The MLP maps `context` features to the location, the log scale and the degrees of
    freedom. The degrees of freedom are `2 + exp(.)`, above 2, so the variance stays finite.
    """

    def __init__(self, features: int, context: int, hidden_features: Sequence[int] = (64, 64), **kwargs) -> None:
        super().__init__()
        self.hyper = MLP(context, 3 * features, hidden_features, **kwargs)

    def forward(self, c: Tensor | None = None) -> Distribution:
        loc, log_scale, log_df = self.hyper(_required(c, "StudentT")).chunk(3, dim=-1)
        return Independent(_StudentT(2 + log_df.exp(), loc, log_scale.exp()), 1)


class Bernoulli(LazyDistribution):
    """Independent Bernoulli variables with probabilities `sigmoid(f(c))` over `features` variables.

    The MLP maps `context` features to the logits. The data must be floating-point zeros and
    ones.
    """

    def __init__(self, features: int, context: int, hidden_features: Sequence[int] = (64, 64), **kwargs) -> None:
        super().__init__()
        self.hyper = MLP(context, features, hidden_features, **kwargs)

    def forward(self, c: Tensor | None = None) -> Distribution:
        logits = self.hyper(_required(c, "Bernoulli"))
        return Independent(_Bernoulli(logits=logits), 1)
