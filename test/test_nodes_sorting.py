# tests clustering with model order selection

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from spikepy.common import TimeSeriesCovE
from spikepy.nodes import BOTMNode

##---TESTS

class TestSortingNodes(ut.TestCase):
    def setUp(self):
        pass

    def testMainSingle(self, do_plot=False):
        import time

        # test setup
        TF = 21
        NC = 2
        spike_proto_sc = sp.cos(sp.linspace(-sp.pi, 3 * sp.pi, TF))
        spike_proto_sc *= sp.hanning(TF)
        scale = sp.linspace(0, 2, TF)
        xi1 = sp.vstack((spike_proto_sc * 5 * scale,
                         spike_proto_sc * 4 * scale)).T
        xi2 = sp.vstack((spike_proto_sc * .5 * scale[::-1],
                         spike_proto_sc * 9 * scale[::-1])).T
        templates = sp.asarray([xi1, xi2])
        LEN = 2000
        noise = sp.randn(LEN, NC)
        ce = TimeSeriesCovE(tf_max=TF, nc=NC)
        ce.update(noise)
        FB = BOTMNode(templates=templates,
                      ce=ce,
                      debug=False,
                      ovlp_taus=None)
        signal = sp.zeros_like(noise)
        NPOS = 4
        POS = [(int(i * LEN / (NPOS + 1)), 100) for i in xrange(1, NPOS + 1)]
        POS.append((100, 2))
        POS.append((150, 2))
        for pos, tau in POS:
            signal[pos:pos + TF] += xi1
            signal[pos + tau:pos + tau + TF] += xi2
        x = sp.ascontiguousarray(signal + noise, dtype=sp.float32)

        print '### constructed spike times ###'
        print POS

        # sort
        tic_o = time.clock()
        FB(x)
        toc_o = time.clock()
        print '### sorting spike times ###'
        print FB.rval

        print '###'
        print 'duration:', toc_o - tic_o

if __name__ == '__main__':
    ut.main()
