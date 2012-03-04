# tests for all the smaller things in the common package

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

from numpy.testing import *
import scipy as sp
from spikepy.common import (INDEX_DTYPE, xi_vs_f, kteo, mteo, sortrows,
                            vec2ten, ten2vec, deprecated, mcvec_from_conc,
                            mcvec_to_conc, xcorr)

##---TESTS-alphabetic-by-file

class TestCommon(ut.TestCase):
    pass


class TestCommonUtil(ut.TestCase):
    def testIndexDtype(self):
        self.assertEqual(INDEX_DTYPE, sp.dtype(sp.int64))

    def testDeprecatedDecorator(self):
        # --- new function
        def sum_many(*args):
            return sum(args)

        # --- old / deprecated function
        @deprecated(sum_many)
        def sum_couple(a, b):
            return a + b

        # --- test
        assert_equal(sum_couple(2, 2), 4)


class TestCommonFuncsFilterutil(ut.TestCase):
    def testXiVsF(self, nc=2):
        xi1 = sp.array([[0, 0, 1, 0, 0]] * nc, dtype=float).T
        xi2 = sp.array([[0, 0, 1, 0, 0]] * nc, dtype=float).T
        xis = sp.asarray([mcvec_to_conc(xi1), mcvec_to_conc(xi2)])
        xvf = xi_vs_f(xis, xis, nc=2)
        assert_equal(xvf.shape, (nc, nc, 2 * xi1.shape[0] - 1))
        assert_equal(xvf.sum(), 8.0)
        assert_equal((xvf != 0.0).sum(), 4)

    def testKTeo(self):
        # TODO: how to test this?!
        pass

    def testMTeo(self):
        # TODO: how to test this?!
        pass


class TestCommonFuncsGeneral(ut.TestCase):
    def testSortrows(self):
        """shamelessly stolen from matlab-docu"""
        data = sp.array([
            [95, 45, 92, 41, 13, 1, 84],
            [95, 7, 73, 89, 20, 74, 52],
            [95, 7, 73, 5, 19, 44, 20],
            [95, 7, 40, 35, 60, 93, 67],
            [76, 61, 93, 81, 27, 46, 83],
            [76, 79, 91, 0, 19, 41, 1],
        ])
        data_sorted = sp.array([
            [76, 61, 93, 81, 27, 46, 83],
            [76, 79, 91, 0, 19, 41, 1],
            [95, 7, 40, 35, 60, 93, 67],
            [95, 7, 73, 5, 19, 44, 20],
            [95, 7, 73, 89, 20, 74, 52],
            [95, 45, 92, 41, 13, 1, 84],
        ])
        assert_equal(sortrows(data), data_sorted)

    def testVec2Ten_Ten2Vec(self, nc=2):
        """multiple observation conversions test"""

        vec_data = sp.array([[1, 2, 3, 4, 5, 6]] * nc)
        ten_data = sp.array([[[1, 4], [2, 5], [3, 6], ]] * nc)
        vec_data_test = ten2vec(ten_data)
        ten_data_test = vec2ten(vec_data, nc)
        assert_equal(vec_data_test, vec_data)
        assert_equal(ten_data_test, ten_data)
        assert_equal(ten2vec(vec2ten(vec_data, nc)), vec_data)
        assert_equal(vec2ten(ten2vec(ten_data), nc), ten_data)

    def testMcvecFromConc_McvecToConc(self, nc=2):
        """single observation conversion test"""

        concv = sp.array([1, 2, 3, 4, 5, 6])
        mcvec = sp.array([[1, 4], [2, 5], [3, 6]])
        concv_test = mcvec_to_conc(mcvec)
        mcvec_test = mcvec_from_conc(concv, nc)
        assert_equal(concv_test, concv)
        assert_equal(mcvec_test, mcvec)
        assert_equal(mcvec_to_conc(mcvec_from_conc(concv, nc)), concv)
        assert_equal(mcvec_from_conc(mcvec_to_conc(mcvec), nc), mcvec)

    def testXcorr(self):
        """shamelessly stolen from matlab-docu"""

        n = 10
        lag_n = 5
        data = sp.ones(n)
        xcorr_test = sp.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 0.9,
             0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        assert_equal(xcorr(sp.zeros(n)).sum(), 0)
        assert_equal(xcorr(data), xcorr_test)
        assert_equal(xcorr(data, sp.zeros(n)), sp.zeros(2 * n - 1))
        assert_equal(xcorr(data, lag=lag_n),
                     xcorr_test[n - lag_n - 1:n + lag_n])
        assert_equal(xcorr(data), xcorr(data, data))
        assert_equal(xcorr(data, data * 2), 2 * xcorr_test)
        assert_equal(xcorr(data, 2 * data), xcorr(data) * 2)

##---MAIN

if __name__ == '__main__':
    ut.main()
