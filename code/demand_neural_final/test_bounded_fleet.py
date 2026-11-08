import itertools
import unittest
import bike_bounded_fleet as model
import numpy as np
from neural_allocation import optimize


class BoundedFleetTests(unittest.TestCase):
    def setUp(self):
        self.caps = np.array([2, 2], dtype=np.int64)
        self.costs = np.array([[[0., -25., -49.], [0., -25., -49.]],
                               [[0., -26., -48.], [0., -24., -49.]]])
        self.truth = np.zeros((2, 2, 24))
        self.truth[:, :, 0] = -3

    def test_exhaustive_minimum_and_zero_tolerance_baseline(self):
        prepared = model.prepare(self.costs, self.caps, self.truth)
        old, _ = optimize(self.costs, self.caps)
        report, allocation = model.select(prepared, 0)
        np.testing.assert_array_equal(old, allocation)
        choices = list(itertools.product(range(3), repeat=2))
        pub_costs = model.scenario_costs(self.truth[:, :, None], np.ones((2, 2, 1)), self.caps)
        last = report['N']
        for tolerance in (0., 0.01, 0.05, 0.1, 0.5, 1.):
            report, allocation = model.select(prepared, tolerance)
            feasible = []
            for n in range(1, prepared['baseline_N'] + 1):
                same_n = [x for x in choices if sum(x) == n]
                def cost(days):
                    return 20*n + np.mean([min(sum(day[i,x[i]] for i in range(2)) for x in same_n) for day in days])
                delta = cost(self.costs) - prepared['predicted_baseline_relative']
                public_delta = cost(pub_costs) - prepared['public_reference_score']
                budget = tolerance * prepared['public_reference_score']
                if delta <= budget + 1e-8 and public_delta <= budget + 1e-8:
                    feasible.append(n)
            self.assertEqual(report['N'], min(feasible))
            self.assertLessEqual(report['N'], last)
            last = report['N']
            self.assertTrue(report['all_smaller_positive_N_infeasible'])
            self.assertTrue((allocation.sum(1) == report['N']).all())

    def test_constant_cost_shift_does_not_change_percentage_budget(self):
        changed = self.costs + np.array([[-10000., 500.], [-3000., -2000.]])[:, :, None]
        original = model.prepare(self.costs, self.caps, self.truth)
        shifted = model.prepare(changed, self.caps, self.truth)
        self.assertLess(original['predicted_baseline_relative'], 0)
        for tolerance in (0., 0.01, 0.1, 1.):
            first, _ = model.select(original, tolerance)
            second, _ = model.select(shifted, tolerance)
            for key in ('N','allowed_daily_score_increase','public_reference_score','predicted_daily_score_increase'):
                self.assertAlmostEqual(first[key], second[key])

    def test_public_actual_bound_is_enforced_and_invalid_rates_rejected(self):
        self.truth[:, :, 0] = -5
        prepared = model.prepare(self.costs, self.caps, self.truth)
        report, _ = model.select(prepared, .1)
        self.assertEqual(report['N'], 4)
        self.assertGreater(report['N_minus_one_public_increase'], report['allowed_daily_score_increase'])
        for rate in (-1., np.nan, np.inf):
            with self.assertRaises(ValueError):
                model.select(prepared, rate)


if __name__ == '__main__':
    unittest.main()
