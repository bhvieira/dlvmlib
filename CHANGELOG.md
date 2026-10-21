# Changelog

All notable changes to dlvmlib are recorded here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-21

### Added

- `VAE`, a deep latent variable model whose encoder, decoder and prior are zuko lazy distributions, with optional covariates and an optional conditional prior.
- `DiagGaussian`, `StudentT` and `Bernoulli`, conditional distributions whose parameters an MLP reads from the context.
- `ELBO`, `STL`, `IWAE` and `DReG`, lower bounds that the encoder and the model both climb, with a closed-form KL term for `ELBO(analytic_kl=True)`.
- `EUBO` and `CUBO`, upper bounds that the encoder descends while the decoder and the prior climb IWAE.
- `SUMO`, an unbiased estimate of `log p(x | c)` by Russian roulette, with the tail `RouletteTail`.
- `VAE.log_marginal` for held-out log-likelihood and `VAE.sample` for ancestral sampling.

[0.1.0]: https://github.com/bhvieira/dlvmlib/releases/tag/v0.1.0
