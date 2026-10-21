"""Gradients: which side each objective trains, and whether its estimator targets the right gradient.

Three families of checks. Path estimators (STL, DReG) must vanish at the exact posterior, where
the reparameterized estimators they replace don't, and must stay unbiased away from it. The
model side of DReG, EUBO and CUBO must be IWAE's, draw for draw, and must estimate the gradient
of the evidence. The encoder side of EUBO and CUBO must descend the closed-form bound. Finally
every objective must train the encoder to the exact posterior, which the Gaussian family
contains, so that is where each divergence is minimized.
"""

from __future__ import annotations

import unittest

import torch

import tests  # noqa: F401  (double precision)
from dlvmlib import CUBO, ELBO, EUBO, IWAE, STL, SUMO, DReG, RouletteTail
from tests import linear_gaussian as lg
from tests.sampling import encoder_parameters, flat, gradient, mean_gradient, model_parameters


class PathEstimators(unittest.TestCase):
    def test_path_estimators_vanish_at_the_posterior(self) -> None:
        torch.manual_seed(0)
        vae, x = lg.build(0.0), lg.data()
        params = encoder_parameters(vae)
        for objective in (STL(1), STL(4), DReG(4)):
            with self.subTest(objective=repr(objective)):
                self.assertLess(gradient(objective, vae, x, params).abs().max().item(), 1e-10)
        for objective in (ELBO(1), IWAE(4)):
            with self.subTest(objective=repr(objective)):
                self.assertGreater(gradient(objective, vae, x, params).abs().max().item(), 1e-3)

    def test_elbo_estimators_are_unbiased(self) -> None:
        torch.manual_seed(0)
        vae, x = lg.build(1.0), lg.data()[3]
        params = encoder_parameters(vae)
        exact = flat(torch.autograd.grad(lg.elbo(vae, x), params))
        errors = {}
        for objective in (ELBO(2), STL(2), ELBO(2, analytic_kl=True)):
            with self.subTest(objective=repr(objective)):
                mean, se = mean_gradient(objective, vae, x, params, reps=1000, batches=20)
                self.assertTrue(((mean - exact).abs() < 4 * se + 1e-12).all(), (mean - exact) / se)
                errors[repr(objective)] = se.norm().item()
        self.assertLess(errors["STL(k=2)"], errors["ELBO(k=2, analytic_kl=False)"])

    def test_dreg_is_unbiased_for_the_iwae_gradient(self) -> None:
        torch.manual_seed(0)
        vae, x = lg.build(1.0), lg.data()[3]
        params = encoder_parameters(vae)
        iwae, iwae_se = mean_gradient(IWAE(4), vae, x, params, reps=1000, batches=20)
        dreg, dreg_se = mean_gradient(DReG(4), vae, x, params, reps=1000, batches=20)
        combined = (iwae_se**2 + dreg_se**2).sqrt()
        self.assertTrue(((dreg - iwae).abs() < 4 * combined).all(), (dreg - iwae) / combined)
        self.assertLess(dreg_se.norm().item(), iwae_se.norm().item())


class ModelSide(unittest.TestCase):
    def test_model_side_matches_the_lower_bound_on_shared_draws(self) -> None:
        vae, x = lg.build(1.0), lg.data()
        params = model_parameters(vae)
        pairs = ((STL(6), ELBO(6)), (DReG(6), IWAE(6)), (EUBO(6), IWAE(6)), (CUBO(6), IWAE(6)))
        for objective, reference in pairs:
            with self.subTest(objective=repr(objective)):
                torch.manual_seed(0)
                ours = gradient(objective, vae, x, params)
                torch.manual_seed(0)
                theirs = gradient(reference, vae, x, params)
                torch.testing.assert_close(ours, theirs, rtol=1e-10, atol=1e-12)

    def test_eubo_model_side_estimates_the_evidence_gradient(self) -> None:
        # The self-normalized estimate is biased by O(1 / k); at k = 1000 that is far below the
        # tolerance, and the standard error covers the rest.
        torch.manual_seed(0)
        vae, x = lg.build(1.0), lg.data()[3]
        params = model_parameters(vae)
        exact = flat(torch.autograd.grad(lg.log_evidence(vae, x), params))
        mean, se = mean_gradient(EUBO(1000), vae, x, params, reps=50, batches=10)
        self.assertTrue(((mean - exact).abs() < 4 * se + 0.01 * exact.abs()).all(), (mean - exact) / se)


class EncoderSide(unittest.TestCase):
    def test_eubo_encoder_descends_the_exact_eubo(self) -> None:
        torch.manual_seed(0)
        vae, x = lg.build(1.0), lg.data()[3]
        params = encoder_parameters(vae)
        exact = flat(torch.autograd.grad(lg.eubo(vae, x), params))
        for mode in ("reparam", "score"):
            with self.subTest(gradient=mode):
                mean, se = mean_gradient(EUBO(1000, gradient=mode), vae, x, params, reps=50, batches=10)
                # The objective's gradient is the ascent direction, so the encoder's part is -dEUBO.
                self.assertTrue(((-mean - exact).abs() < 4 * se + 0.01 * exact.abs()).all(), (-mean - exact) / se)

    def test_cubo_encoder_descends_the_exact_cubo(self) -> None:
        torch.manual_seed(0)
        vae, x = lg.build(1.0), lg.data()[3]
        params = encoder_parameters(vae)
        for n in (2.0, 3.0):
            with self.subTest(n=n):
                exact = flat(torch.autograd.grad(lg.cubo(vae, x, n), params))
                # The rescaling in CHIVI's Algorithm 1 changes the length of each step but not
                # its direction, so the direction is what's compared.
                ours = -gradient(CUBO(200000, n=n), vae, x.unsqueeze(0), params)
                cosine = torch.nn.functional.cosine_similarity(ours, exact, dim=0)
                self.assertGreater(cosine.item(), 0.999)


class Training(unittest.TestCase):
    def test_every_objective_trains_the_encoder_to_the_posterior(self) -> None:
        # CUBO gets 64 draws: with few draws its encoder collapses (see its docstring). The
        # gradients are clipped, as the SUMO paper does for its encoder; the step size decays
        # tenfold, so the estimators whose noise persists at the optimum settle too.
        objectives = (
            ELBO(1),
            ELBO(1, analytic_kl=True),
            STL(1),
            IWAE(4),
            DReG(4),
            EUBO(8),
            EUBO(8, gradient="score"),
            CUBO(64),
            SUMO(m=2, tail=RouletteTail(alpha=8)),
        )
        x = lg.data().repeat(16, 1)
        target = flat(encoder_parameters(lg.build(0.0))).detach()
        for objective in objectives:
            with self.subTest(objective=repr(objective)):
                torch.manual_seed(0)
                vae = lg.build(1.0)
                params = encoder_parameters(vae)
                optimizer = torch.optim.Adam(params, lr=0.01)
                schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 1500))
                for _ in range(1500):
                    optimizer.zero_grad()
                    (-objective(vae, x).mean()).backward()
                    torch.nn.utils.clip_grad_norm_(params, 10.0)
                    optimizer.step()
                    schedule.step()
                self.assertLess((flat(params) - target).abs().max().item(), 0.03)

    def test_path_estimators_land_on_the_posterior(self) -> None:
        # At the posterior the STL and DReG gradients are exactly zero, so training stops there.
        x = lg.data().repeat(16, 1)
        target = flat(encoder_parameters(lg.build(0.0))).detach()
        for objective in (STL(1), DReG(4)):
            with self.subTest(objective=repr(objective)):
                torch.manual_seed(0)
                vae = lg.build(1.0)
                params = encoder_parameters(vae)
                optimizer = torch.optim.Adam(params, lr=0.01)
                schedule = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 1500))
                for _ in range(1500):
                    optimizer.zero_grad()
                    (-objective(vae, x).mean()).backward()
                    optimizer.step()
                    schedule.step()
                self.assertLess((flat(params) - target).abs().max().item(), 1e-4)

    def test_cubo_encoder_collapses_with_few_draws(self) -> None:
        # Documents the failure in CUBO's docstring: dividing by the largest weight makes each
        # step follow the k-draw log estimate, which falls without bound as q narrows.
        torch.manual_seed(0)
        vae = lg.build(1.0)
        params = encoder_parameters(vae)
        optimizer = torch.optim.Adam(params, lr=0.01)
        for _ in range(300):
            optimizer.zero_grad()
            (-CUBO(8)(vae, lg.data().repeat(16, 1)).mean()).backward()
            optimizer.step()
        encoder = vae.encoder
        assert isinstance(encoder, lg.AffineEncoder)
        self.assertTrue((encoder.log_t < -2.0).all())


if __name__ == "__main__":
    unittest.main()
