"""Score calibration, safe-domain constraints, and integer-optimality checks."""
import itertools
import unittest
import bike_score_consensus as model
import numpy as np


class ConsensusTests(unittest.TestCase):
    def test_score_offset_cancels_and_known_contrast_is_fitted(self):
        a=np.array([[[0.,-50.,-60.],[0.,0.,0.]],[[0.,-40.,-50.],[0.,0.,0.]]])
        b=np.array([[[0.,10.,20.],[0.,0.,0.]],[[0.,20.,30.],[0.,0.,0.]]])
        first=np.array([[2,0],[2,0]]);second=np.array([[1,0],[1,0]])
        weight,fit=model.fit_pair(a,b,first,second,100.,80.)
        self.assertAlmostEqual(fit['fitted_contrast'],-20.)
        c0=a+np.array([[[500.],[20.]],[[100.],[900.]]])
        c1=b+np.array([[[50.],[210.]],[[10.],[90.]]])
        shifted,_=model.fit_pair(c0,c1,first,second,100.,80.)
        self.assertAlmostEqual(weight,shifted)

    def test_restriction_preserves_exact_days_and_reference_feasibility(self):
        costs=np.zeros((3,2,5));first=np.array([[0,3],[1,2],[0,3]]);second=np.array([[2,1],[2,1],[2,1]])
        limited=model.restrict_to_reference_range(costs,first,second,known_days=[1])
        self.assertTrue(np.isinf(limited[0,0,3]))
        self.assertTrue(np.isfinite(limited[1]).all())
        for x in [first,second]:self.assertTrue(np.isfinite(limited[np.arange(3)[:,None],np.arange(2),x]).all())

    def test_consensus_dp_matches_exhaustive_common_fleet_search(self):
        caps=np.array([2,2])
        a=np.array([[[0.,-60.,-70.],[0.,-20.,-25.]],[[0.,-40.,-50.],[0.,-10.,-20.]]])
        b=np.array([[[0.,-20.,-10.],[0.,-10.,-15.]],[[0.,-10.,-5.],[0.,-30.,-35.]]])
        first=np.array([[2,1],[2,1]]);second=np.array([[1,1],[1,1]])
        d0=model.predicted_score(a,second)-model.predicted_score(a,first)
        d1=model.predicted_score(b,second)-model.predicted_score(b,first)
        target=.5*(d0+d1)
        allocation,costs,_=model.solve(a,b,first,second,100.,100.+target,caps)
        best=float('inf')
        rows=list(itertools.product(range(3),repeat=2))
        for r0 in rows:
            for r1 in rows:
                if sum(r0)!=sum(r1) or sum(r0)==0:continue
                x=np.asarray([r0,r1])
                if np.isfinite(costs[np.arange(2)[:,None],np.arange(2),x]).all():best=min(best,model.predicted_score(costs,x))
        self.assertAlmostEqual(model.predicted_score(costs,allocation),best)


if __name__=='__main__':unittest.main()
