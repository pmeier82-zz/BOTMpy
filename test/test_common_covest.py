# tests for covariance estimation

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from spikepy.common import TimeSeriesCovE

##---TESTS

class TestCovarianceEstimator(ut.TestCase):
    def setUp(self):
        self.dlen = 250000
        self.tf = 65
        self.nc = 4
        self.white_noise = sp.randn(self.dlen, self.nc)
        self.CE = TimeSeriesCovE(tf_max=self.tf, nc=self.nc)
        self.CE.new_chan_set((1, 2))
        self.CE.update(self.white_noise)

    def testTrivial(self):
        p_4_20 = {'tf':20, 'chan_set':(0, 1, 2, 3)}
        C_4_20 = self.CE.get_cmx(**p_4_20)
        self.assertTupleEqual(C_4_20.shape, (4 * 20, 4 * 20 ))
        assert_equal(C_4_20, C_4_20.T)

        p_2_10 = {'tf':10, 'chan_set':(0, 1)}
        C_2_10 = self.CE.get_cmx(**p_2_10)
        self.assertTupleEqual(C_2_10.shape, (2 * 10, 2 * 10 ))
        assert_equal(C_2_10, C_2_10.T)

    def testInverse(self):
        p_4_20 = {'tf':20, 'chan_set':(0, 1, 2, 3)}
        C_4_20 = self.CE.get_cmx(**p_4_20)
        iC_4_20 = self.CE.get_icmx(**p_4_20)
        should_be_eye80 = sp.dot(C_4_20, iC_4_20)
        assert_almost_equal(should_be_eye80, sp.eye(80), decimal=5)

        p_2_10 = {'tf':10, 'chan_set':(0, 1)}
        C_2_10 = self.CE.get_cmx(**p_2_10)
        iC_2_10 = self.CE.get_icmx(**p_2_10)
        should_be_eye20 = sp.dot(C_2_10, iC_2_10)
        assert_almost_equal(should_be_eye20, sp.eye(20), decimal=5)

##---MAIN

if __name__ == '__main__':
    ut.main()
