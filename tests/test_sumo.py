"""SUMO: the roulette tail, the telescoping sum, and unbiasedness.

The tail must follow Luo et al.'s Eq. 10 and its sampler must reproduce it. With every weight
set to one, the roulette sum telescopes to IWAE on all `m + K` draws, which pins the indexing.
Unbiasedness is checked where it matters: for `m > 1` too, where an off-by-one in the weights
would still pass at `m = 1`, and against IWAE_m's own bias, so the check has the power to fail.
"""

from __future__ import annotations

import math
import unittest

import torch

import tests  # noqa: F401  (double precision)
from dlvmlib import IWAE, SUMO, RouletteTail, log_mean_exp, sumo_estimate
from tests import linear_gaussian as lg
from tests.sampling import mean_value


class Tail(unittest.TestCase):
    def test_tail_follows_equation_10(self) -> None:
        tail = RouletteTail()
        j = torch.tensor([1, 2, 79, 80, 81, 100])
        expected = torch.tensor([1, 1 / 2, 1 / 79, 1 / 80, 0.9 / 80, 0.9**20 / 80])
        torch.testing.assert_close(tail.survival(j), expected)
        self.assertAlmostEqual(tail.mean(), sum(1 / j for j in range(1, 80)) + 1 / 8)

    def test_sampled_terms_follow_the_tail(self) -> None:
        torch.manual_seed(0)
        tail = RouletteTail()
        n = 1_000_000
        terms = tail.terms(1 - torch.rand(n))
        self.assertGreaterEqual(terms.min().item(), 1)
        for j in (1, 2, 3, 10, 79, 80, 81, 100):
            with self.subTest(j=j):
                p = tail.survival(torch.tensor(j)).item()
                observed = (terms >= j).double().mean().item()
                self.assertLess(abs(observed - p), 5 * math.sqrt(p * (1 - p) / n) + 1e-12)
        mean = terms.double().mean().item()
        self.assertLess(abs(mean - tail.mean()), 5 * terms.double().std().item() / math.sqrt(n))


class Estimate(unittest.TestCase):
    def test_unit_weights_telescope_to_iwae(self) -> None:
        log_w = torch.randn(9, 5)
        for m in (1, 3, 8):
            with self.subTest(m=m):
                torch.testing.assert_close(sumo_estimate(log_w, m, torch.ones(9 - m)), log_mean_exp(log_w))

    def test_mismatched_weights_are_refused(self) -> None:
        # A single weight would otherwise broadcast across every roulette term.
        log_w = torch.randn(9, 5)
        for m, terms in ((3, 1), (3, 7), (0, 9), (10, 0)):
            with self.subTest(m=m, terms=terms), self.assertRaises(ValueError):
                sumo_estimate(log_w, m, torch.ones(terms))

    def test_sumo_is_unbiased(self) -> None:
        torch.manual_seed(0)
        vae, x = lg.build(1.0), lg.data()[3]
        evidence = lg.log_evidence(vae, x).item()
        for m in (1, 3):
            with self.subTest(m=m):
                mean, se = mean_value(SUMO(m=m, tail=RouletteTail(alpha=8)), vae, x, reps=500, batches=400)
                self.assertLess(abs(mean.item() - evidence), 4 * se.item())
                iwae, _ = mean_value(IWAE(m), vae, x, reps=500, batches=20)
                self.assertGreater(evidence - iwae.item(), 10 * se.item())


if __name__ == "__main__":
    unittest.main()
