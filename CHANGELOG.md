# Changelog

All notable changes to dlvmlib are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-21

### Added

- IWAE-STL, as `IWAE(k, gradient="stl")`. It's biased for `k > 1` (Tucker et al., 2019).

### Changed

- `ELBO` and `IWAE` take a `gradient` argument that chooses the encoder's gradient estimator: `"reparam"` (the default), `"stl"`, or `"dreg"` for `IWAE` only.

### Removed

- `STL` and `DReG`. Use `ELBO(k, gradient="stl")` for ELBO-STL and `IWAE(k, gradient="dreg")` for IWAE-DReG.

### Fixed

- The docs no longer present IWAE-DReG as IWAE with `log q` frozen, which is IWAE-STL.
- The docs' attributions to the EUBO, CUBO and SUMO papers: the EUBO constant-scaling example, why CHIVI rescales its gradient, and which SUMO choices come from the paper.

## [0.1.0] - 2026-10-21

### Added

- `VAE`, a deep latent variable model whose encoder, decoder and prior are zuko lazy distributions, with optional covariates and an optional conditional prior.
- `DiagGaussian`, `StudentT` and `Bernoulli`, conditional distributions whose parameters an MLP reads from the context.
- `ELBO`, `STL`, `IWAE` and `DReG`, lower bounds that the encoder and the model both climb, with a closed-form KL term for `ELBO(analytic_kl=True)`.
- `EUBO` and `CUBO`, upper bounds that the encoder descends while the decoder and the prior climb IWAE.
- `SUMO`, an unbiased estimate of `log p(x | c)` by Russian roulette, with the tail `RouletteTail`.
- `VAE.log_marginal` for held-out log-likelihood and `VAE.sample` for ancestral sampling.

[0.2.0]: https://github.com/bhvieira/dlvmlib/releases/tag/v0.2.0
[0.1.0]: https://github.com/bhvieira/dlvmlib/releases/tag/v0.1.0
