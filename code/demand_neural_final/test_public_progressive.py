import unittest
import update_public_progressive as update
import numpy as np


class ProgressivePublicTests(unittest.TestCase):
    def test_newly_revealed_D24_is_not_a_test_day(self):
        public=np.array([8,9,22,23])
        days=update.evaluation_days(public)
        np.testing.assert_array_equal(days,np.arange(24,30))
        self.assertEqual(len(np.intersect1d(public,days)),0)
        self.assertEqual((days+1).tolist(),[25,26,27,28,29,30])

    def test_all_revealed_future_days_are_excluded(self):
        public=np.array([8,9,22,23,26,29])
        np.testing.assert_array_equal(update.evaluation_days(public),[24,25,27,28])
        with self.assertRaises(ValueError):
            update.evaluation_days(np.arange(30))

    def test_predecessor_observations_are_aligned_by_day_not_position(self):
        public=np.array([8,9,22,23])
        truth=np.array([10.,20.,30.,40.])[:,None,None]
        np.testing.assert_array_equal(update.subset_truth(public,truth,[22,8,9]).ravel(),[30.,10.,20.])
        with self.assertRaises(ValueError):
            update.subset_truth(public,truth,[8,9,25])


if __name__=='__main__':
    unittest.main()
