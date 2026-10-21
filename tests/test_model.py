"""The model's wiring: shapes, covariates, frozen evaluation, zuko flows in every role, and evaluation.

The objectives lean on three promises from `VAE`: draws come first in every shape, covariates
reach every part that should read them, and frozen evaluation passes gradients to its inputs
and to nothing else. Zuko's flows must fill any role, including through the frozen path, which
calls them with detached parameters.
"""

from __future__ import annotations

import unittest

import torch
import zuko

import tests  # noqa: F401  (double precision)
from dlvmlib import EUBO, IWAE, VAE, Bernoulli, DiagGaussian, DReG, StudentT
from tests import linear_gaussian as lg
from tests.sampling import encoder_parameters, model_parameters

FEATURES, LATENT, COVARIATES, BATCH = 5, 2, 3, 4


def flow_vae(conditional_prior: bool = False) -> VAE:
    context = COVARIATES if conditional_prior else 0
    return VAE(
        encoder=zuko.flows.NSF(LATENT, context=COVARIATES + FEATURES),
        decoder=zuko.flows.MAF(FEATURES, context=COVARIATES + LATENT),
        prior=zuko.flows.MAF(LATENT, context=context),
        conditional_prior=conditional_prior,
    )


class Shapes(unittest.TestCase):
    def test_draws_come_first(self) -> None:
        vae, x = lg.build(1.0), lg.data()
        for frozen in (False, True):
            with self.subTest(frozen=frozen):
                z, log_q = vae.draw(x, k=5, frozen=frozen)
                self.assertEqual(z.shape, (5, 4, lg.LATENT))
                self.assertEqual(log_q.shape, (5, 4))
                self.assertEqual(vae.log_joint(x, z).shape, (5, 4))

    def test_samples(self) -> None:
        self.assertEqual(lg.build().sample(6).shape, (6, lg.FEATURES))
        c = torch.randn(BATCH, COVARIATES)
        for conditional_prior in (False, True):
            with self.subTest(conditional_prior=conditional_prior):
                self.assertEqual(flow_vae(conditional_prior).sample(6, c).shape, (6, BATCH, FEATURES))


class Covariates(unittest.TestCase):
    def test_covariates_reach_every_part(self) -> None:
        torch.manual_seed(0)
        vae = flow_vae(conditional_prior=True)
        x, c = torch.randn(BATCH, FEATURES), torch.randn(BATCH, COVARIATES)
        (-IWAE(3)(vae, x, c).mean()).backward()
        for name, p in vae.named_parameters():
            with self.subTest(parameter=name):
                grad = p.grad
                assert grad is not None
                self.assertGreater(grad.abs().sum().item(), 0.0)
        z = torch.randn(3, BATCH, LATENT)
        with torch.no_grad():
            self.assertFalse(torch.allclose(vae.log_joint(x, z, c), vae.log_joint(x, z, c + 1)))
            self.assertFalse(torch.allclose(vae.latent_prior(c).log_prob(z), vae.latent_prior(c + 1).log_prob(z)))

    def test_conditional_prior_needs_covariates(self) -> None:
        vae = VAE(DiagGaussian(LATENT, FEATURES), DiagGaussian(FEATURES, LATENT), zuko.flows.MAF(LATENT, 1), True)
        with self.assertRaises(ValueError):
            vae.latent_prior()


class Frozen(unittest.TestCase):
    def test_frozen_joint_passes_gradient_only_to_the_latents(self) -> None:
        vae, x = lg.build(1.0), lg.data()
        z = torch.randn(3, 4, lg.LATENT, requires_grad=True)
        vae.log_joint(x, z, frozen=True).sum().backward()
        grad = z.grad
        assert grad is not None
        self.assertGreater(grad.abs().sum().item(), 0.0)
        self.assertTrue(all(p.grad is None for p in model_parameters(vae)))
        self.assertTrue(all(p.requires_grad for p in vae.parameters()))

    def test_frozen_draw_keeps_only_the_path_term(self) -> None:
        vae, x = lg.build(1.0), lg.data()
        encoder = vae.encoder
        assert isinstance(encoder, lg.AffineEncoder)
        z, log_q = vae.draw(x, k=3, frozen=True)
        score = vae.posterior(x).log_prob(z.detach())
        (path,) = torch.autograd.grad(log_q.sum(), [encoder.log_t])
        (scored,) = torch.autograd.grad(score.sum(), [encoder.log_t])
        # With z = mu + t eps, the path term of d log q / d log t is -eps^2 and the score term
        # is eps^2 - 1; together they make the total derivative, -1 per draw.
        eps = ((z - x @ encoder.A.T - encoder.d) / encoder.log_t.exp()).detach()
        torch.testing.assert_close(path, -(eps**2).sum((0, 1)))
        torch.testing.assert_close(path + scored, torch.full_like(path, -12.0))


class Families(unittest.TestCase):
    def test_zuko_flows_fill_every_role(self) -> None:
        torch.manual_seed(0)
        vae = flow_vae()
        x, c = torch.randn(BATCH, FEATURES), torch.randn(BATCH, COVARIATES)
        (-DReG(3)(vae, x, c).mean()).backward()
        for name, p in vae.named_parameters():
            with self.subTest(parameter=name):
                grad = p.grad
                assert grad is not None
                self.assertTrue(torch.isfinite(grad).all())

    def test_families_have_the_declared_shapes(self) -> None:
        c = torch.randn(BATCH, 3)
        for family in (DiagGaussian(FEATURES, 3), StudentT(FEATURES, 3), Bernoulli(FEATURES, 3)):
            with self.subTest(family=type(family).__name__):
                distribution = family(c)
                self.assertEqual(distribution.batch_shape, (BATCH,))
                self.assertEqual(distribution.event_shape, (FEATURES,))
                with self.assertRaises(ValueError):
                    family()

    def test_a_discrete_encoder_trains_by_the_score_form(self) -> None:
        torch.manual_seed(0)

        def fair_coins(probs: torch.Tensor) -> torch.distributions.Distribution:
            return torch.distributions.Independent(torch.distributions.Bernoulli(probs), 1)

        vae = VAE(
            Bernoulli(LATENT, FEATURES),
            DiagGaussian(FEATURES, LATENT),
            zuko.lazy.UnconditionalDistribution(fair_coins, torch.full((LATENT,), 0.5), buffer=True),
        )
        x = torch.randn(BATCH, FEATURES)
        with self.assertRaises(ValueError):
            IWAE(3)(vae, x)
        (-EUBO(3, gradient="score")(vae, x).mean()).backward()
        for p in encoder_parameters(vae):
            grad = p.grad
            assert grad is not None
            self.assertGreater(grad.abs().sum().item(), 0.0)


class Evaluation(unittest.TestCase):
    def test_log_marginal_is_exact_at_the_exact_posterior(self) -> None:
        vae, x = lg.build(0.0), lg.data()
        torch.testing.assert_close(vae.log_marginal(x, k=7, chunk=3), lg.log_evidence(vae, x).detach())

    def test_log_marginal_approaches_the_evidence(self) -> None:
        torch.manual_seed(0)
        vae, x = lg.build(1.0), lg.data()
        torch.testing.assert_close(
            vae.log_marginal(x, k=20000, chunk=5000), lg.log_evidence(vae, x).detach(), rtol=0, atol=0.02
        )


if __name__ == "__main__":
    unittest.main()
