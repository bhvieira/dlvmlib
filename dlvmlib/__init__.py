"""Deep latent variable models on zuko distributions.

A `VAE` holds three zuko lazy distributions: an encoder `q(z | x, c)`, a decoder `p(x | z, c)`
and a prior `p(z)` or `p(z | c)`. The objectives train it from `k` importance-weighted draws of
the encoder:

    ELBO, STL     lower bounds averaged over draws; STL drops the encoder's score-function term
    IWAE, DReG    the importance-weighted bound; DReG's encoder gradient is doubly reparameterized
    EUBO, CUBO    upper bounds that the encoder descends while the model climbs IWAE
    SUMO          an unbiased estimate of log p(x | c); the encoder descends its variance

Each objective returns a tensor of shape `batch`. Its value is the objective's estimate and its
gradient trains every part, so `(-objective(vae, x, c).mean()).backward()` is a complete
backward pass.
"""

from dlvmlib.families import Bernoulli, DiagGaussian, StudentT
from dlvmlib.model import VAE
from dlvmlib.objectives import CUBO, ELBO, EUBO, IWAE, STL, Bound, DReG, log_mean_exp, self_normalized, two_sided
from dlvmlib.sumo import SUMO, RouletteTail, sumo_estimate

__all__ = [
    "CUBO",
    "ELBO",
    "EUBO",
    "IWAE",
    "STL",
    "SUMO",
    "VAE",
    "Bernoulli",
    "Bound",
    "DReG",
    "DiagGaussian",
    "RouletteTail",
    "StudentT",
    "log_mean_exp",
    "self_normalized",
    "sumo_estimate",
    "two_sided",
]
