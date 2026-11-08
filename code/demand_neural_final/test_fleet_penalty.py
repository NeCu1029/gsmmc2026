import itertools
import unittest
import bike_fleet_penalty as model
import numpy as np
from neural_allocation import optimize


class FleetPenaltyTests(unittest.TestCase):
    def setUp(self):
        self.caps = np.array([2, 3], dtype=np.int64)
        self.costs = np.array([[[80., 30., 0., np.inf], [130., 60., 20., 0.]],
                               [[85., 32., 0., np.inf], [125., 65., 18., 0.]]])

    def test_zero_alpha_reproduces_original_optimizer(self):
        original, objective = optimize(self.costs, self.caps)
        [(row, new)] = model.frontier(self.costs, self.caps, [0.])
        np.testing.assert_array_equal(new, original)
        self.assertAlmostEqual(row['relative_objective'], objective)

    def test_matches_exhaustive_search_and_monotonic_fleet(self):
        alphas = [0., 5., 20., 50., 200.]
        entries = model.frontier(self.costs, self.caps, alphas)
        allocations = list(itertools.product(range(3), range(4)))
        for alpha, (row, result) in zip(alphas, entries):
            exact = []
            for n in range(1, 6):
                choices = [x for x in allocations if sum(x) == n]
                daily = [min(sum(day[i, x[i]] for i in range(2)) for x in choices) for day in self.costs]
                exact.append(((20 + alpha) * n + np.mean(daily), n))
            objective, n = min(exact)
            self.assertEqual(row['N'], n)
            self.assertAlmostEqual(row['relative_objective'], objective)
            self.assertTrue((result.sum(1) == n).all())
        self.assertTrue(all(a[0]['N'] >= b[0]['N'] for a, b in zip(entries, entries[1:])))
        self.assertLess(entries[-1][0]['N'], entries[0][0]['N'])

    def test_rejects_invalid_penalties_and_costs(self):
        for alpha in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                model.frontier(self.costs, self.caps, [alpha])
        broken = self.costs.copy()
        broken[0, 0, 1] = np.nan
        with self.assertRaises(ValueError):
            model.frontier(broken, self.caps)


if __name__ == '__main__':
    unittest.main()
