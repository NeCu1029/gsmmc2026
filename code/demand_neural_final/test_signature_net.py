import copy
import unittest
import bike_signature_net as model
import numpy as np
import torch
from test_neural_models import toy_data


class SignatureTests(unittest.TestCase):
    def test_nonpublic_labels_cannot_change_public_features(self):
        data = toy_data()
        public = data.city.net_flow[:1].numpy().copy()
        first = model.make_inputs(data, public)
        changed = copy.deepcopy(data)
        changed.city.net_flow[1] += 5000
        changed.counts[1] += 5000
        second = model.make_inputs(changed, public)
        for name in ('signature', 'anchor', 'spread'):
            torch.testing.assert_close(getattr(first, name), getattr(second, name))

    def test_single_station_prediction_and_gradient(self):
        data = toy_data()
        inputs = model.make_inputs(data, data.city.net_flow[:1].numpy())
        target = model.transfer.increments(data.city.net_flow.numpy(), data.capacity)
        net, _ = model.fit(inputs, target, np.array([1]), np.array([0, 1]), 42, epochs=1)
        pred = model.predict(net, inputs, np.array([1]), np.array([2]))
        self.assertEqual(pred.shape, (1, 1, 3))
        self.assertTrue(np.isfinite(pred).all())

    def test_zero_blend_uses_public_empirical_costs(self):
        anchor = np.array([[1., 2., 3.]])
        pred = np.array([[[8., 9., 10.]], [[2., 3., 4.]]])
        result = model.blend(pred, anchor, np.array([False, True]), 0., 1.)
        np.testing.assert_array_equal(result[0], anchor)
        np.testing.assert_array_equal(result[1], pred[1])


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    unittest.main()
