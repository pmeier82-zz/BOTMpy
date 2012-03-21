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

    def testMainSingle(self, do_plot=True):
        from spikeplot import plt, mcdata
        import time

        # test setup
        C_SIZE = 410
        TF = 21
        NC = 2
        xi1 = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi, TF))] * NC).T * 2
        xi2 = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi, TF))] * NC).T * 5
        templates = sp.asarray([xi1, xi2])
        LEN = 200000
        noise = sp.randn(LEN, NC)
        ce = TimeSeriesCovE(tf_max=TF, nc=NC)
        ce.update(noise)
        FB = BOTMNode(templates=templates,
                      ce=ce,
                      adapt_templates=15,
                      learn_noise=False,
                      debug=False,
                      spk_pr=1e-6,
                      ovlp_taus=None)
        signal = sp.zeros_like(noise)
        NPOS = 4
        POS = [(int(i * LEN / (NPOS + 1)), 100) for i in xrange(1, NPOS + 1)]
        POS.append((100, 2))
        POS.append((120, 2))
        print POS
        for pos, tau in POS:
            signal[pos:pos + TF] += xi1
            signal[pos + tau:pos + tau + TF] += xi2
        x = sp.ascontiguousarray(signal + noise, dtype=sp.float32)

        # sort
        tic_o = time.clock()
        FB(x)
        toc_o = time.clock()
        print 'duration:', toc_o - tic_o

        # plotting
        if do_plot:
            ev = {}
            for u in xrange(FB.nfilter):
                ev[u] = (FB.bank[u].xi, FB.rval[u])
            fouts = FB._disc
            print ev
            ovlp_meth = 'sic'
            if FB._ovlp_taus is not None:
                ovlp_meth = 'och'
            print 'overlap method:', ovlp_meth
            mcdata(x, events=ev, other=fouts,
                   title='overlap method: %s' % ovlp_meth)
            FB.plot_xvft()
            plt.show()

if __name__ == '__main__':
    ut.main()
