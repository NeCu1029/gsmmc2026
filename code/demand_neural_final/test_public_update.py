import unittest
import update_public_ensemble as update
from test_neural_models import toy_data
import numpy as np
import pandas as pd
import torch


class PublicUpdateTests(unittest.TestCase):
    def test_new_public_day_is_exact_without_modifying_neural_weights(self):
        data = toy_data()
        data.city.net_flow = torch.cat([data.city.net_flow, data.city.net_flow[:1].clone()])
        data.city.net_flow[2, 0, 0] = -20
        data.city.temporal = torch.cat([data.city.temporal, data.city.temporal[:1]])
        data.city.dates = pd.date_range('2026-06-01', periods=3)
        data.city.off_day = np.array([False, True, False])
        data.counts = torch.stack([(-data.city.net_flow).clamp_min(0)+2, data.city.net_flow.clamp_min(0)+2], dim=-1)
        old_days, new_days, sites = np.array([0,1]), np.arange(3), np.arange(3)
        old_truth, new_truth = data.city.net_flow[:2].numpy(), data.city.net_flow.numpy()
        inputs = update.neural.make_inputs(data, old_truth)
        torch.manual_seed(71)
        p = update.neural.inventory.InventoryNet(data.similarity.shape[-1], update.neural.inventory.day_features(data).shape[-1])
        p.fit_scales(data, old_days, sites)
        s = update.neural.SignatureNet(inputs.signature.shape[-1], update.neural.inventory.day_features(data).shape[-1])
        s.fit_scales(inputs, old_days, sites)
        prior = {'model_type':'public_marginal_inventory_v1','public_days':[0,1],'ns':data.similarity.shape[-1],
                 'nt':update.neural.inventory.day_features(data).shape[-1],'config':{'weekday_weight':1.,'offday_weight':.5},'states':[p.state_dict()]}
        signature = {'model_type':'public_signature_v2','public_days':[0,1],'ns':inputs.signature.shape[-1],
                     'nt':update.neural.inventory.day_features(data).shape[-1],'config':{'weekday_blend':.5,'holiday_blend':.5},'states':[s.state_dict()]}
        before = [{k:v.clone() for k,v in checkpoint['states'][0].items()} for checkpoint in (prior,signature)]
        old = update.infer(data,prior,signature,old_days,old_truth,new_days,sites)
        new = update.infer(data,prior,signature,new_days,new_truth,new_days,sites)
        exact = update.neural.transfer.increments(new_truth,data.capacity)
        np.testing.assert_array_equal(new,exact)
        self.assertFalse(np.allclose(old[2],new[2]))
        self.assertEqual(update.neural.make_inputs(data,new_truth).signature.shape,inputs.signature.shape)
        for checkpoint,previous in zip((prior,signature),before):
            for k,v in checkpoint['states'][0].items():
                torch.testing.assert_close(v,previous[k])

    def test_original_public_day_cannot_be_silently_removed(self):
        prior={'model_type':'public_marginal_inventory_v1','public_days':[8,9]}
        signature={'model_type':'public_signature_v2','public_days':[8,9]}
        update.check_public_protocol(prior,signature,np.array([8,9,22]))
        with self.assertRaises(ValueError):
            update.check_public_protocol(prior,signature,np.array([8,22]))


if __name__=='__main__':
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    unittest.main()
