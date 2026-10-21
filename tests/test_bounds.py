"""Values of the objectives: exactness at the posterior, ordering, and convergence to the evidence.

At the exact posterior every log weight equals `log p(x)`, so every objective must return the
evidence to rounding, whatever its reduction. Away from it, the reductions order themselves on
shared draws: the mean is at most the log-mean-exp, which is at most the self-normalized mean
and the chi estimate. Expectations are checked against the closed forms of `linear_gaussian`.
"""

from __future__ import annotations

import unittest

import torch
import zuko

import tests  # noqa: F401  (double precision)
from dlvmlib import CUBO, ELBO, EUBO, IWAE, STL, SUMO, DReG
from tests import linear_gaussian as lg
from tests.sampling import mean_value


class Exactness(unittest.TestCase):
    def test_every_objective_returns_the_evidence_at_the_posterior(self) -> None:
        torch.manual_seed(0)
        vae, x = lg.build(0.0), lg.data()
        evidence = lg.log_evidence(vae, x).detach()
        objectives = (ELBO(3), STL(3), IWAE(3), DReG(3), EUBO(3), EUBO(3, "score"), CUBO(3), SUMO(m=2))
        for objective in objectives:
            with self.subTest(objective=repr(objective)):
                torch.testing.assert_close(objective(vae, x).detach(), evidence, rtol=0, atol=1e-10)


class Ordering(unittest.TestCase):
    def test_reductions_order_on_shared_draws(self) -> None:
        vae, x = lg.build(1.0), lg.data()
        values = {}
        for objective in (ELBO(8), IWAE(8), EUBO(8), CUBO(8)):
            torch.manual_seed(0)
            values[type(objective).__name__] = objective(vae, x).detach()
        self.assertTrue((values["ELBO"] <= values["IWAE"]).all())
        self.assertTrue((values["IWAE"] <= values["EUBO"]).all())
        self.assertTrue((values["IWAE"] <= values["CUBO"]).all())

    def test_iwae_climbs_toward_the_evidence(self) -> None:
        torch.manual_seed(0)
        vae, x = lg.build(1.0), lg.data()[3]
        evidence = lg.log_evidence(vae, x).item()
        means = []
        for k in (1, 4, 32):
            mean, se = mean_value(IWAE(k), vae, x, reps=2000, batches=10)
            self.assertLess(mean.item(), evidence + 4 * se.item())
            means.append(mean.item())
        self.assertLess(means[0], means[1])
        self.assertLess(means[1], means[2])
        self.assertLess(evidence - means[2], (evidence - means[0]) / 4)


class Expectations(unittest.TestCase):
    def test_elbo_estimates_agree_with_the_closed_form(self) -> None:
        torch.manual_seed(0)
        vae, x = lg.build(1.0), lg.data()[3]
        exact = lg.elbo(vae, x).item()
        for objective in (ELBO(1), ELBO(4), STL(4), ELBO(1, analytic_kl=True)):
            with self.subTest(objective=repr(objective)):
                mean, se = mean_value(objective, vae, x, reps=2000, batches=10)
                self.assertLess(abs(mean.item() - exact), 4 * se.item())

    def test_analytic_kl_needs_a_closed_form(self) -> None:
        vae = lg.build()
        vae.encoder = zuko.flows.NSF(lg.LATENT, context=lg.FEATURES)
        with self.assertRaises(TypeError):
            ELBO(1, analytic_kl=True)(vae, lg.data())


class Arguments(unittest.TestCase):
    def test_invalid_arguments_are_refused(self) -> None:
        refused = (
            lambda: ELBO(0),
            lambda: IWAE(0),
            lambda: EUBO(1),
            lambda: EUBO(4, gradient="paper"),  # ty: ignore[invalid-argument-type]
            lambda: CUBO(1),
            lambda: CUBO(4, n=1.0),
            lambda: SUMO(m=0),
        )
        for make in refused:
            with self.subTest(), self.assertRaises(ValueError):
                make()


if __name__ == "__main__":
    unittest.main()
