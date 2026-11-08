import copy
import unittest
from unittest.mock import patch
import bike_neural_final as forecast
import bike_inventory_net as inventory
import bike_public_transfer as transfer
import numpy as np
import pandas as pd
import torch


def toy_data():
    rng = np.random.default_rng(71)
    net = forecast.base.tensor(rng.integers(-3, 4, (2, 3, 24)))
    counts = torch.stack([(-net).clamp_min(0) + 2, net.clamp_min(0) + 2], dim=-1)
    city = forecast.base.City(pd.Index(["A001", "A002", "A003"]), pd.date_range("2026-06-01", periods=2),
                              forecast.base.tensor(rng.normal(size=(3, 12))), torch.eye(3),
                              forecast.base.tensor(rng.normal(size=(2, 24, 27))), np.array([False, True]), net)
    return forecast.Data(city, counts, np.array([2, 3, 2]), rng.normal(size=(3, 24)), np.zeros((2, 5)))


class NeuralModelTests(unittest.TestCase):
    def test_public_correction_respects_day_type_and_zero_weight(self):
        pred = np.zeros((2, 3, 3))
        public = np.ones((2, 3, 3))
        truth = public + 4
        off = np.array([False, True])
        public_off = np.array([False, False])
        result = transfer.adjusted(pred, public, truth, off, public_off, 0.5, 0)
        np.testing.assert_array_equal(result[0], np.full((3, 3), 2))
        np.testing.assert_array_equal(result[1], np.zeros((3, 3)))
        np.testing.assert_array_equal(pred, np.zeros((2, 3, 3)))

    def test_scaling_uses_training_labels_only(self):
        data = toy_data()
        changed = copy.deepcopy(data)
        changed.city.net_flow[1] += 1000
        changed.counts[1] += 2000
        changed.city.temporal[1] += 1000
        for cls in (forecast.ForecastNet,):
            first = cls(12, 27, "graph_mlp")
            second = cls(12, 27, "graph_mlp")
            first.fit_scales(data, np.array([0]), np.array([0, 1]))
            second.fit_scales(changed, np.array([0]), np.array([0, 1]))
            for key in ("y_mean", "y_scale", "t_mean", "t_scale", "s_mean", "s_scale"):
                torch.testing.assert_close(getattr(first, key), getattr(second, key))

    def test_networks_have_finite_training_gradients(self):
        data = toy_data()
        for architecture in forecast.ARCHITECTURES:
            model, _ = forecast.fit(data, np.array([0]), np.array([0, 1]), architecture, 71, epochs=1)
            result = forecast.predict_one(model, data, np.array([1]), np.array([2]))
            self.assertEqual(result.shape, (1, 1, 24))
            self.assertTrue(np.isfinite(result).all())
        costs = forecast.scenario_costs(data.city.net_flow.numpy()[:, :, None, :].astype(float), np.ones((2, 3, 1)), data.capacity)
        target = np.diff(np.where(np.isfinite(costs), costs, 0), axis=-1)
        model, _ = inventory.fit(data, target, np.array([0]), np.array([0, 1]), 71, epochs=1)
        self.assertTrue(np.isfinite(inventory.predict(model, data, np.array([1]), np.array([2]))).all())

    def test_marginal_costs_preserve_exact_optimum(self):
        data = toy_data()
        costs = forecast.scenario_costs(data.city.net_flow.numpy()[:, :, None, :].astype(float), np.ones((2, 3, 1)), data.capacity)
        increments = np.diff(np.where(np.isfinite(costs), costs, 0), axis=-1)
        relative = inventory.cost_table(increments, data.capacity)
        x, score = forecast.optimize(costs, data.capacity)
        y, reduced_score = forecast.optimize(relative, data.capacity)
        np.testing.assert_array_equal(x, y)
        self.assertAlmostEqual(score, reduced_score + costs[:, :, 0].sum(1).mean())

    def test_scenarios_preserve_neural_mean(self):
        data = toy_data()
        rng = np.random.default_rng(7)
        mean = rng.normal(size=(2, 3, 24))
        paths = rng.normal(size=(2, 3, 4, 24))
        weights = np.broadcast_to(np.array([0.1, 0.2, 0.3, 0.4]), (2, 3, 4)).copy()
        real = forecast.scenario_costs
        seen = []
        def record(s, w, c):
            seen.append((s * w[:, :, :, None]).sum(2))
            return real(s, w, c)
        with patch.object(forecast, "scenario_costs", side_effect=record):
            for risk in forecast.RISKS:
                forecast.allocation(mean, paths, weights, risk, data.capacity)
        for result in seen:
            np.testing.assert_allclose(result, mean, atol=1e-12)


if __name__ == "__main__":
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    unittest.main()
