"""Regression checks for the mean-forecast default and checkpoint compatibility."""
import inspect
import unittest

import torch
import forecast_net as model


class MeanForecastTests(unittest.TestCase):
    def test_default_loss_is_symmetric_mse(self):
        weight = inspect.signature(model.fit).parameters["over_weight"].default
        self.assertEqual(weight, 1.0)
        prediction = torch.tensor([-2.0, 2.0], requires_grad=True)
        actual = torch.zeros(2)
        loss = model.objective(prediction, actual, weight)
        self.assertEqual(loss.item(), 4.0)
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.tensor([-2.0, 2.0]))

    def test_average_is_preferred_to_downward_shift(self):
        truth = torch.tensor([-5.0, 5.0])
        at_mean = model.objective(torch.zeros(2), truth, model.DEFAULT_OVER_WEIGHT)
        below_mean = model.objective(torch.full((2,), -2.5), truth, model.DEFAULT_OVER_WEIGHT)
        self.assertLess(at_mean.item(), below_mean.item())

    def test_asymmetric_checkpoint_requires_explicit_matching_setting(self):
        saved = {"model_type": "direct_net_gnn_lstm", "target": "returns - rentals",
                 "k": model.K, "loss_config": {"over_weight": 3.0}}
        with self.assertRaisesRegex(ValueError, "over_weight"):
            model.validate_checkpoint(saved, model.DEFAULT_OVER_WEIGHT)
        model.validate_checkpoint(saved, 3.0)
        saved["loss_config"]["over_weight"] = 1.0
        model.validate_checkpoint(saved, model.DEFAULT_OVER_WEIGHT)
        saved["loss_config"] = {}
        with self.assertRaises(ValueError):
            model.validate_checkpoint(saved, model.DEFAULT_OVER_WEIGHT)


if __name__ == "__main__":
    unittest.main()
