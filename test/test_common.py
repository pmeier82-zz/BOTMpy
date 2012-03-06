# tests for all the smaller things in the common package

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from spikepy.common import (INDEX_DTYPE, xi_vs_f, kteo, mteo, sortrows,
                            vec2ten, ten2vec, deprecated, mcvec_from_conc,
                            mcvec_to_conc, xcorr, shifted_matrix_sub,
                            dict_list_to_ndarray, dict_sort_ndarrays, get_idx)

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

    def testXcorr1(self):
        """testing the xcorr function, trivial test of functionality"""

        n = 10
        lag_n = 5
        data = sp.ones(n)
        xcorr_test = sp.array([
            1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 9., 8., 7., 6., 5., 4.,
            3., 2., 1.])
        assert_equal(xcorr(sp.zeros(n)).sum(), 0)
        assert_equal(xcorr(data), xcorr_test)
        assert_equal(xcorr(data, sp.zeros(n)), sp.zeros(2 * n - 1))
        assert_equal(xcorr(data, lag=lag_n),
                     xcorr_test[n - lag_n - 1:n + lag_n])
        assert_equal(xcorr(data), xcorr(data, data))
        assert_equal(xcorr(data, data * 2), 2 * xcorr_test)
        assert_equal(xcorr(data, 2 * data), xcorr(data) * 2)

    def testXcorr2(self):
        """testing the xcorr function, sine acorr against matlab reference"""

        n = 10
        lag_n = 5
        data = sp.array([
            0.00000000e+00, 6.42787610e-01, 9.84807753e-01, 8.66025404e-01,
            3.42020143e-01, -3.42020143e-01, -8.66025404e-01, -9.84807753e-01,
            -6.42787610e-01, -2.44929360e-16])
        xcorr_test = sp.array([
            2.22044605e-16, 4.44089210e-16, -4.13175911e-01, -1.26604444e+00,
            -2.08318711e+00, -2.14542968e+00, -9.83955557e-01, 1.19459271e+00,
            3.44719999e+00, 4.50000000e+00, 3.44719999e+00, 1.19459271e+00,
            -9.83955557e-01, -2.14542968e+00, -2.08318711e+00,
            -1.26604444e+00, -4.13175911e-01, 4.44089210e-16, 1.11022302e-16])
        assert_almost_equal(xcorr(data), xcorr_test)
        assert_almost_equal(xcorr(data, lag=lag_n),
                            xcorr_test[n - lag_n - 1:n + lag_n])

    def testShiftedMatrixSub(self):
        """test for shifted matrix subtraction"""

        at, nsub, ns, nc = 25, 10, 100, 2
        data = sp.zeros((ns, nc))
        data[at:at + nsub] = 1.0
        sub = sp.ones((10, nc))
        sub_test = sp.zeros_like(data)
        assert_equal(shifted_matrix_sub(data, sub, at), sub_test)
        # TODO: complete for edge cases

    def testDictListToNdarray(self):
        """test for recursive dict list to ndarray converter"""

        d = {'str':'a string',
             'dict':{'a':'dict',
                     'arr10':range(10)},
             'arr10':range(10)}
        dict_list_to_ndarray(d)
        self.assertIsInstance(d['arr10'], sp.ndarray)
        self.assertIsInstance(d['dict']['arr10'], sp.ndarray)

    def testDictSortNdarrays(self):
        """test for recursive dict ndarray sorter"""

        d = {'str':'a string',
             'dict':{'a':'dict',
                     'arr10':sp.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])},
             'arr10':sp.array([5, 6, 7, 8, 9, 0, 1, 2, 3, 4])}
        dict_sort_ndarrays(d)
        assert_equal(d['arr10'], sp.arange(10))
        assert_equal(d['dict']['arr10'], sp.arange(10))

    def testGetIdx(self):
        """test for index sets"""

        idxs = [0, 2, 3, 4, 7]
        assert_equal(get_idx(idxs, append=False), 1)
        assert_equal(get_idx(idxs, append=True), 8)

##---MAIN

if __name__ == '__main__':
    ut.main()
