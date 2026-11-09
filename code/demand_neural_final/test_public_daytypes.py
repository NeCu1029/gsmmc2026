import unittest
from types import SimpleNamespace
from unittest.mock import patch
import update_public_daytypes as update
import numpy as np


class DayTypePublicTests(unittest.TestCase):
    def setUp(self):
        self.data=SimpleNamespace(city=SimpleNamespace(off_day=np.array([False,True,False,True])),capacity=np.array([2]))
        self.prior={'model_type':'public_marginal_inventory_v1','public_days':[0],
                    'config':{'weekday_weight':1.,'offday_weight':.5}}
        self.signature={'model_type':'public_signature_v2','public_days':[0]}
        self.actual=np.array([[[10.,10.]],[[40.,40.]]])

    def reference(self,data,prior,signature,public_days,truth,days,sites):
        self.assertEqual(public_days.tolist(),[0])
        return np.repeat(np.where(data.city.off_day[days],20.,10.)[:,None,None],2,axis=-1)

    def test_weekend_residual_changes_only_unobserved_weekend(self):
        with patch.object(update.core.base,'infer',side_effect=self.reference),patch.object(update.core.base.neural.transfer,'increments',return_value=self.actual):
            result=update.infer_daytypes(self.data,self.prior,self.signature,np.array([0,1]),np.zeros((2,1,24)),np.array([2,3]),np.array([0]))
        np.testing.assert_array_equal(result[:,0],[[10.,10.],[30.,30.]])

    def test_public_weekend_uses_full_actual_cost_not_half_correction(self):
        with patch.object(update.core.base,'infer',side_effect=self.reference),patch.object(update.core.base.neural.transfer,'increments',return_value=self.actual):
            result=update.infer_daytypes(self.data,self.prior,self.signature,np.array([0,1]),np.zeros((2,1,24)),np.arange(4),np.array([0]))
        np.testing.assert_array_equal(result[:2],self.actual)
        np.testing.assert_array_equal(result[2:,0],[[10.,10.],[30.,30.]])

    def test_weekday_only_history_is_exactly_backward_compatible(self):
        with patch.object(update.core.base,'infer',side_effect=self.reference),patch.object(update.core.base.neural.transfer,'increments',side_effect=AssertionError('No new-type correction expected')):
            result=update.infer_daytypes(self.data,self.prior,self.signature,np.array([0]),np.zeros((1,1,24)),np.array([2,3]),np.array([0]))
        np.testing.assert_array_equal(result[:,0],[[10.,10.],[20.,20.]])


if __name__=='__main__':
    unittest.main()
