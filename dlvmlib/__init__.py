"""Deep latent variable models on zuko distributions.

A `VAE` holds three zuko lazy distributions: an encoder `q(z | x, c)`, a decoder `p(x | z, c)`
and a prior `p(z)` or `p(z | c)`. The objectives train it from `k` importance-weighted draws of
the encoder:

    ELBO          the ELBO averaged over draws, with the ELBO-STL encoder gradient
    IWAE          the importance-weighted bound, with the IWAE-STL and IWAE-DReG encoder gradients
    EUBO, CUBO    upper bounds that the encoder descends while the model climbs IWAE
    SUMO          an unbiased estimate of log p(x | c); the encoder descends its variance

Each objective returns a tensor of shape `batch`. Its value is the objective's estimate and its
gradient trains every part, so `(-objective(vae, x, c).mean()).backward()` is a complete
backward pass.
"""

from dlvmlib.families import Bernoulli, DiagGaussian, StudentT
from dlvmlib.model import VAE
from dlvmlib.objectives import CUBO, ELBO, EUBO, IWAE, Bound, log_mean_exp, self_normalized, two_sided
from dlvmlib.sumo import SUMO, RouletteTail, sumo_estimate

__all__ = [
    "CUBO",
    "ELBO",
    "EUBO",
    "IWAE",
    "SUMO",
    "VAE",
    "Bernoulli",
    "Bound",
    "DiagGaussian",
    "RouletteTail",
    "StudentT",
    "log_mean_exp",
    "self_normalized",
    "sumo_estimate",
    "two_sided",
]
