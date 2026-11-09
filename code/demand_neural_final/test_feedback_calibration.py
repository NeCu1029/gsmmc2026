"""Physical recurrence and hidden-target isolation checks."""
import unittest
from types import SimpleNamespace
import bike_feedback_calibration as model
import run_feedback_refinement as run
import numpy as np
import pandas as pd


class FeedbackTests(unittest.TestCase):
    def test_official_penalty_and_sign(self):
        # rent five, return ten: stock 2 -> shortage 3 -> overflow 6.
        truth=np.zeros((1,1,24));truth[0,0,:2]=[-5,10]
        metrics=model.observed_metrics(truth,np.array([[2]]),np.array([4]))
        np.testing.assert_array_equal(metrics,[[3,9,6,46]])
        self.assertEqual(20*2+6*metrics[0,1]+5*metrics[0,2]+metrics[0,3],170)

    def test_hidden_target_labels_do_not_affect_scenarios(self):
        rng=np.random.default_rng(5)
        dates=pd.date_range('2026-06-01',periods=30)
        off=np.asarray(dates.dayofweek>=5)
        source_counts=rng.poisson(2,size=(30,5,24,2)).astype(float)
        source={'net':source_counts[...,1]-source_counts[...,0],'counts':source_counts,
                'capacity':np.array([4,6,8,5,7]),'similarity':rng.normal(size=(5,24)),
                'off':off,'dates':dates.strftime('%Y-%m-%d').to_numpy(str)}
        target=SimpleNamespace(city=SimpleNamespace(dates=dates,off_day=off),
                               capacity=np.array([4,6]),similarity=rng.normal(size=(2,24)))
        public=np.array([8,9,12]);counts=rng.poisson(2,size=(3,2,24,2)).astype(float)
        first=model.scenarios(source,target,np.arange(2),public,counts)
        target.city.net_flow=np.full((30,2,24),np.nan)
        target.counts=np.full((30,2,24,2),1e9)
        second=model.scenarios(source,target,np.arange(2),public,counts)
        for a,b in zip(first,second):np.testing.assert_array_equal(a,b)
        np.testing.assert_allclose(first[1].sum(-1),1,rtol=0,atol=1e-12)

    def test_screenshot_daily_and_monthly_reconciliation(self):
        feedback=run.screenshot_feedback()
        self.assertEqual(feedback.shape,(30,4))
        self.assertEqual(feedback[13,3],3682)
        self.assertEqual(feedback[28,0],2255)


if __name__=='__main__':unittest.main()
