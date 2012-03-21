# tests ringbuffer structures

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from spikepy.common.mcfilter.mcfilter_cy import (
    _mcfilter_cy32, _mcfilter_cy64, _mcfilter_hist_cy32, _mcfilter_hist_cy64)
from spikepy.common.mcfilter.mcfilter_py import (
    _mcfilter_py, _mcfilter_hist_py)

##---TESTS

class TestMcFilter(ut.TestCase):
    def testHistoryCy32(self):
        """test history item"""
        tf = 3
        nc = 2
        data = sp.randn(100, nc).astype(sp.float32)
        filt = sp.ones((tf, nc), dtype=sp.float32)
        hist = sp.zeros((tf - 1, nc), dtype=sp.float32)
        fout, hist = _mcfilter_hist_cy32(data, filt, hist)
        assert_equal(hist, data[-(tf - 1):])

    def testHistoryCy64(self):
        """test history item"""
        tf = 3
        nc = 2
        data = sp.randn(100, nc).astype(sp.float64)
        filt = sp.ones((tf, nc), dtype=sp.float64)
        hist = sp.zeros((tf - 1, nc), dtype=sp.float64)
        fout, hist = _mcfilter_hist_cy64(data, filt, hist)
        assert_equal(hist, data[-(tf - 1):])

    def testPyVsCyOnesCy32(self):
        """test python and cython, float"""
        tf = 3
        nc = 2
        data = sp.ones((20, nc), dtype=sp.float32)
        filt = sp.ones((tf, nc), dtype=sp.float32)
        hist = sp.ones((tf - 1, nc), dtype=sp.float32)
        fopy, hopy = _mcfilter_hist_py(data, filt, hist)
        focy, hocy = _mcfilter_hist_cy32(data, filt, hist)
        assert_almost_equal(fopy, focy)

    def testPyVsCyOnesCy64(self):
        """test python and cython, double"""
        tf = 3
        nc = 2
        data = sp.ones((20, nc), dtype=sp.float64)
        filt = sp.ones((tf, nc), dtype=sp.float64)
        hist = sp.ones((tf - 1, nc), dtype=sp.float64)
        fopy, hopy = _mcfilter_hist_py(data, filt, hist)
        focy, hocy = _mcfilter_hist_cy64(data, filt, hist)
        assert_almost_equal(fopy, focy)

    def testPyVsCyRandnCy32(self):
        """test python and cython"""
        tf = 3
        nc = 2
        data = sp.randn(20, nc).astype(sp.float32)
        filt = sp.ones((tf, nc), dtype=sp.float32)
        hist_py = sp.ones((tf - 1, nc), dtype=sp.float32)
        hist_cy = sp.ones((tf - 1, nc), dtype=sp.float32)
        fopy, hopy = _mcfilter_hist_py(data, filt, hist_py)
        focy, hocy = _mcfilter_hist_cy32(data, filt, hist_cy)
        assert_almost_equal(fopy, focy)

    def testPyVsCyRandnCy64(self):
        """test python and cython"""
        tf = 3
        nc = 2
        data = sp.randn(20, nc).astype(sp.float64)
        filt = sp.ones((tf, nc), dtype=sp.float64)
        hist_py = sp.ones((tf - 1, nc), dtype=sp.float64)
        hist_cy = sp.ones((tf - 1, nc), dtype=sp.float64)
        fopy, hopy = _mcfilter_hist_py(data, filt, hist_py)
        focy, hocy = _mcfilter_hist_cy64(data, filt, hist_cy)
        assert_almost_equal(fopy, focy)

    def testStepsCy32(self):
        tf = 3
        nc = 2
        data = sp.vstack([sp.concatenate(
            [sp.arange(1, 4)] * 5)] * 2).T.astype(sp.float32)
        filt = sp.ones((tf, nc), dtype=sp.float32) / float(tf)
        hist_py = sp.ones((tf - 1, nc), dtype=sp.float32)
        hist_cy = sp.ones((tf - 1, nc), dtype=sp.float32)
        fopy, hopy = _mcfilter_hist_py(data, filt, hist_py)
        focy, hocy = _mcfilter_hist_cy32(data, filt, hist_cy)
        assert_almost_equal(fopy, focy)

    def testStepsCy64(self):
        tf = 3
        nc = 2
        data = sp.vstack([sp.concatenate(
            [sp.arange(1, 4)] * 5)] * 2).T.astype(sp.float64)
        filt = sp.ones((tf, nc), dtype=sp.float64) / float(tf)
        hist_py = sp.ones((tf - 1, nc), dtype=sp.float64)
        hist_cy = sp.ones((tf - 1, nc), dtype=sp.float64)
        fopy, hopy = _mcfilter_hist_py(data, filt, hist_py)
        focy, hocy = _mcfilter_hist_cy64(data, filt, hist_cy)
        assert_almost_equal(fopy, focy)

    def testDataConcatenationCy32(self):
        data = sp.zeros((100, 1), dtype=sp.float32)
        data[sp.arange(0, 100, 10)] = 1.0
        filt = sp.zeros((5, 1), dtype=sp.float32)
        filt[2] = 1.0
        hist = sp.zeros((4, 1), dtype=sp.float32)
        fout = _mcfilter_hist_cy32(data, filt, hist)[0]
        cut = sp.floor(5.0 / 2)
        assert_equal(data[:-cut], sp.array([fout[cut:]]).T)

    def testDataConcatenationCy64(self):
        data = sp.zeros((100, 1), dtype=sp.float64)
        data[sp.arange(0, 100, 10)] = 1.0
        filt = sp.zeros((5, 1), dtype=sp.float64)
        filt[2] = 1.0
        hist = sp.zeros((4, 1), dtype=sp.float64)
        fout = _mcfilter_hist_cy64(data, filt, hist)[0]
        cut = sp.floor(5.0 / 2)
        assert_equal(data[:-cut], sp.array([fout[cut:]]).T)

    def testMcfilterRecoveryPy(self):
        data = sp.zeros((100, 1), dtype=sp.float64)
        data[sp.arange(0, 100, 10)] = 1.0
        filt = sp.zeros((5, 1), dtype=sp.float64)
        filt[2] = 1.0
        fout = _mcfilter_py(data, filt)
        self.assertTupleEqual(data.shape, (fout.shape[0], 1))
        assert_equal(data, sp.array([fout]).T)

"""
def mcfilter_hist_py_test(inp=None, plot=False):
    if inp is None:
        # test setup
        TF = 10
        NC = 2
        xi = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi,
        TF))] * NC).T * 5
        LEN = 2000
        noise = sp.randn(LEN, NC)

        # build signal
        signal = sp.zeros_like(noise)
        NPOS = 3
        POS = [int(i * LEN / (NPOS + 1)) for i in xrange(1, NPOS + 1)]
        for i in xrange(NPOS):
            signal[POS[i]:POS[i] + TF] += xi
        x = signal + noise
    else:
        x, xi = inp
        TF, NC = xi.shape
    ns = x.shape[0]

    step = 200
    chunks = [x[i * step:(i + 1) * step] for i in xrange(ns / step)]
    fouts = []
    h = None
    for chunk in chunks:
        r, h = _mcfilter_hist_py(chunk, xi, h)
        fouts.append(r)

    if plot:
        from spikeplot import mcdata

        other = sp.atleast_2d(sp.concatenate(fouts)).T
        other = sp.vstack([other, sp.zeros((int(TF / 2 - 1), 1))])[
                int(TF / 2 - 1):, :]
        mcdata(x, other=other)


def mcfilter_hist_c_test(inp=None, plot=False):
    if _mcfilter_hist_cy is None:
        print 'No clib loaded! returning'
        return

    if inp is None:
        # test setup
        TF = 10
        NC = 2
        xi = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi,
        TF))] * NC).T * 5
        LEN = 2000
        noise = sp.randn(LEN, NC)

        # build signal
        signal = sp.zeros_like(noise)
        NPOS = 3
        POS = [int(i * LEN / (NPOS + 1)) for i in xrange(1, NPOS + 1)]
        for i in xrange(NPOS):
            signal[POS[i]:POS[i] + TF] += xi
        x = signal + noise
    else:
        x, xi = inp
    ns = x.shape[0]

    step = 200
    chunks = [x[i * step:(i + 1) * step] for i in xrange(ns / step)]
    fouts = []
    h = sp.zeros((xi.shape[0], xi.shape[1]), dtype=sp.float32)
    #    r = sp.array([0] * ns, dtype=sp.float32)
    for chunk in chunks:
        r, h = _mcfilter_hist_cy(chunk, sp.ascontiguousarray(xi), h)
        fouts.append(r)

    if plot:
        from spikeplot import mcdata

        mcdata(x, other=sp.atleast_2d(sp.concatenate(fouts)).T)


def gen_data(ns=200000, nc=4, tf=65):
    # test setup
    xi = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi, tf))] * nc).T * 7

    signal = sp.randn(ns, nc).astype(sp.float32)
    # build signal
    pos = [50 + i  for i in xrange(1, ns, 4 * tf - 50)]
    if pos[-1] + tf > ns:
        pos.pop(-1)
    for i in xrange(len(pos)):
        signal[pos[i]:pos[i] + tf, :] += xi

    return signal, tf, nc, xi.astype(sp.float32)

if __name__ == '__main__':
    # generate some data
    sig, tf, nc, xi = gen_data(64000)

    # python conventional test
    mcfilter_hist_py_test((sig, xi), plot=True)
    mcfilter_hist_c_test((sig, xi), plot=True)

#    import cProfile
#    cProfile.run('mcfilter_hist_py_test((sig, xi), plot=False)')
#    cProfile.run('mcfilter_hist_c_test((sig, xi), plot=False)')
"""

if __name__ == '__main__':
    ut.main()
