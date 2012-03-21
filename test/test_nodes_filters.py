# tests clustering with model order selection

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from spikepy.common import (TimeSeriesCovE, mcfilter, mcvec_to_conc,
                            mcvec_from_conc)
from spikepy.nodes import MatchedFilterNode, NormalisedMatchedFilterNode

##---TESTS

class TestFilterNodes(ut.TestCase):
    def setUp(self):
        self.tf = 10
        self.nc = 2
        self.xi = sp.vstack(
            [sp.arange(self.tf).astype(sp.float32)] * self.nc).T * 0.5
        self.len = 1000
        self.pos = [int(i * self.len / 4.0) for i in xrange(1, 4)]
        self.noise = sp.randn(self.len, self.nc)
        self.ce = TimeSeriesCovE(tf_max=self.tf, nc=self.nc)
        self.ce.update(self.noise)

    def testFilters(self):
        mf_h = MatchedFilterNode(self.tf, self.nc, self.ce)
        mf_h.append_xi_buf(self.xi, recalc=True)
        nmf_h = NormalisedMatchedFilterNode(self.tf, self.nc, self.ce)
        nmf_h.append_xi_buf(self.xi, recalc=True)
        f = sp.dot(mcvec_to_conc(self.xi), self.ce.get_icmx(tf=self.tf))
        nf = sp.dot(f, mcvec_to_conc(self.xi))
        f = mcvec_from_conc(f, nc=self.nc)
        assert_equal(mf_h.f, f)
        assert_equal(nmf_h.f, f / nf)

        """
        # build signals
        signal = sp.zeros_like(noise)
        for i in xrange(3):
            signal[POS[i]:POS[i] + TF] = xi
        x = signal + noise
        late = int(TF / 2 - 1)
        pad = sp.zeros(late)
        y_h_out = mf_h(x)
        y_h = sp.concatenate([y_h_out[late:], pad])

        #  plot
        from spikeplot import plt

        plt.plot(mcfilter(x, mf_h.f), label='mcfilter (scipy.correlate)',
                 color='r')
        plt.plot(y_h + .02, label='mcfilter_hist (py/c)', color='g')
        plt.plot(signal + 5)
        plt.plot(x + 15)
        plt.legend()
        plt.show()
        """

if __name__ == '__main__':
    ut.main()
