# dlvmlib

Deep latent variable models on [zuko](https://github.com/probabilists/zuko) distributions, and the objectives that train them.

A `VAE` holds three zuko lazy distributions: an encoder `q(z | x, c)`, a decoder `p(x | z, c)`, and a prior `p(z)` or `p(z | c)`.
The encoder and the decoder can each be a Gaussian MLP, a normalizing flow or a mixture, and the prior can be a flow, a mixture or a fixed distribution such as `zuko.lazy.UnconditionalDistribution(DiagNormal, ...)`.
A Gaussian MLP needs a context, so it can be the prior only when the prior is conditional.
A mixture can't be reparameterized, so as the encoder it trains only by `EUBO(gradient="score")`.
`c` holds optional covariates, which the encoder reads as `cat(c, x)`, the decoder as `cat(c, z)`, and a conditional prior as `c`.
Seven objectives train the model from `k` importance-weighted draws of the encoder: ELBO, STL, IWAE, DReG, EUBO, CUBO and SUMO.

## Install

```bash
pip install dlvmlib
```

pip installs PyTorch from PyPI.
To choose a CUDA or CPU build, install PyTorch first by following [its instructions](https://pytorch.org/get-started/locally/).

## Quickstart

```python
import torch
import zuko

from dlvmlib import VAE, DiagGaussian, DReG

features, latent, covariates = 64, 8, 3
vae = VAE(
    encoder=DiagGaussian(latent, context=covariates + features),     # q(z | x, c)
    decoder=zuko.flows.NSF(features, context=covariates + latent),   # p(x | z, c)
    prior=zuko.flows.MAF(latent),                                    # p(z)
)
objective = DReG(k=16)
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

for x, c in loader:
    optimizer.zero_grad()
    (-objective(vae, x, c).mean()).backward()
    optimizer.step()

log_px = vae.log_marginal(x, c, k=5000)   # held-out log-likelihood, IWAE with 5000 draws
samples = vae.sample(10, c)               # (10, batch, features)
```

Every objective returns a tensor of shape `batch`.
Its value is the objective's estimate, for logging, and its gradient is the objective's gradient estimator.
One `backward()` and one optimizer train the encoder, the decoder and the prior, even for the objectives that train them in opposite directions.

## The objectives

Every objective draws `z_1..z_k ~ q(z | x, c)` and forms the log weights `log w_i = log p(x, z_i | c) - log q(z_i | x, c)`.

### Lower bounds: ELBO, STL, IWAE, DReG

These four share one computation and differ in two switches: how the `k` log weights are reduced, and whether `log q` is evaluated with the encoder's parameters frozen.

|                  | `log q` follows the encoder | `log q` with the encoder frozen |
|------------------|-----------------------------|---------------------------------|
| mean over `k`    | `ELBO(k)`                   | `STL(k)`                        |
| log-mean-exp     | `IWAE(k)`                   | `DReG(k)`                       |

Freezing the encoder inside `log q` removes the score-function term from the encoder's gradient and leaves only the path through `z` (Roeder et al., 2017).
STL is unbiased for the ELBO's gradient at any `k`.
DReG additionally scales the gradient reaching each draw by its normalized weight, which makes it unbiased for IWAE's gradient (Tucker et al., 2019).
At the exact posterior both gradients are exactly zero, so they stop there, while the plain reparameterized estimators keep their noise.
`ELBO(k, analytic_kl=True)` computes the KL term in closed form when torch registers one, as between `DiagGaussian` and a diagonal Gaussian prior.

### Upper bounds and SUMO: EUBO, CUBO, SUMO

For these three, the encoder and the model follow different gradients.

| Objective | Value | Encoder | Decoder and prior |
|---|---|---|---|
| `EUBO(k)` | `sum_i w_hat_i log w_i`, the self-normalized EUBO (Ji & Shen, 2019, Eq. 6) | descends it | climb IWAE |
| `CUBO(k, n=2)` | `(1/n) log mean_i w_i^n` (Dieng et al., 2017) | descends `exp(n CUBO)` | climb IWAE |
| `SUMO(m=1)` | the Russian-roulette estimate of `log p(x \| c)` (Luo et al., 2020) | descends `SUMO^2`, its variance | climb SUMO |

The encoder descends an upper bound to tighten it.
That can't lower the evidence, because `log p(x | c)` doesn't depend on the encoder.
The model can't descend it, since that would lower the evidence too, and it can't ascend it either: the bound is `log p(x | c) + KL(p(z | x, c) || q)`, so the model could raise it by moving its posterior away from `q`.
So the model climbs IWAE on the same draws.
With the weights held fixed, that is the model gradient of the EUBO estimate itself, `sum_i w_hat_i d log p(x, z_i) / d theta`, which estimates `d log p(x | c) / d theta` by Fisher's identity.

`two_sided` routes the two gradients.
The model's term is computed from detached draws, and the encoder's term with the decoder and prior frozen, so the decoder and prior are evaluated twice.

## Decisions

### EUBO: the gradient of the bound, not Eq. 5

`EUBO(gradient="reparam")`, the default, differentiates the self-normalized estimate through its weights.
`EUBO(gradient="score")` is the paper's score form, `-sum_i w_hat_i d log q(z_i) / d phi` at fixed draws, and it also trains an encoder that can't be reparameterized, such as `Bernoulli`.
The paper's Eq. 5 isn't offered.
It leaves the factor `log w + 1` off the `log q` term, so it isn't the gradient of the bound, and it changes when the joint is scaled by a constant, as the paper's own minibatch scaling `N / S` does.
On a one-dimensional Gaussian with `log p(x) = 5`, it gives `dU/dsigma = -2.33` where the exact value is `+0.34`.

### CUBO needs many draws

CUBO follows CHIVI's Eq. 3 and Algorithm 1, which divide each datapoint's gradient by `exp(n max_i log w_i)` to prevent overflow.
That rescaling depends on the draws, so each step follows the gradient of the biased `k`-draw estimate, which falls without bound as `q` narrows.
On the test model the encoder collapses at `k = 8` and converges at `k = 64`.
The draws needed grow with the latent dimension.

### SUMO draws one `K` per call

As in the paper's Algorithm 1, one number of terms `K` is drawn per call and shared by the batch, and every datapoint's estimate stays unbiased.
`RouletteTail` is the paper's Eq. 10: `P(K >= j) = 1 / j` below `alpha = 80` and geometric beyond, about 5 terms on average.
The paper clips the encoder's gradients, because the gradient of `SUMO^2` is occasionally very large.

## Develop

```bash
uv sync
uv run pre-commit install
uv run python -m unittest discover -s tests -t .
```

`pre-commit install` makes every commit pass `ruff check`, `ruff format --check` and `ty check` first.
The tests take a minute or two on a CPU and run in double precision, because they compare against closed forms to tight tolerances.

## Tests

The oracle is a linear-Gaussian model whose posterior is diagonal, so the affine Gaussian encoder contains it exactly, and the evidence, the ELBO, the EUBO and the CUBO are closed form.

- At the exact posterior every objective returns `log p(x)` to rounding, and the STL and DReG encoder gradients vanish.
- ELBO, STL and analytic-KL ELBO are unbiased for the closed-form ELBO gradient; DReG agrees with IWAE's gradient and has lower variance.
- The model side of STL, DReG, EUBO and CUBO equals ELBO's or IWAE's draw for draw, and EUBO's model side matches the closed-form evidence gradient.
- EUBO's encoder gradient, in both forms, matches the closed-form EUBO gradient; CUBO's direction matches the closed-form CUBO gradient at large `k`.
- SUMO's tail and sampler follow Eq. 10, its sum telescopes to IWAE with unit weights, and it is unbiased for `m = 1` and `m = 3`.
- Every objective trains the encoder to the exact posterior; CUBO with 8 draws collapses, as documented.

## Layout

```
dlvmlib/
  model.py        VAE: wiring, draws, frozen evaluation, sampling, held-out log-likelihood
  families.py     DiagGaussian, StudentT, Bernoulli: conditional distributions from an MLP
  objectives.py   ELBO, STL, IWAE, DReG on one base; EUBO, CUBO on two_sided
  sumo.py         SUMO, RouletteTail, sumo_estimate
tests/
  linear_gaussian.py   the closed-form oracle
```

## References

- Burda, Grosse & Salakhutdinov (2016). Importance weighted autoencoders.
- Dieng, Tran, Ranganath, Paisley & Blei (2017). Variational inference via χ upper bound minimization.
- Ji & Shen (2019). Stochastic variational inference via upper bound.
- Kingma & Welling (2014). Auto-encoding variational Bayes.
- Luo, Beatson, Norouzi, Zhu, Duvenaud, Adams & Chen (2020). SUMO: unbiased estimation of log marginal probability for latent variable models.
- Roeder, Wu & Duvenaud (2017). Sticking the landing: simple, lower-variance gradient estimators for variational inference.
- Tucker, Lawson, Gu & Maddison (2019). Doubly reparameterized gradient estimators for Monte Carlo objectives.
