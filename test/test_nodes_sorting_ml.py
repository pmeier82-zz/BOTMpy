# tests clustering with model order selection

##---IMPORTS

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from scipy.io import loadmat
from spikepy.common import TimeSeriesCovE, mcvec_from_conc
from spikepy.nodes import BOTMNode

##---TESTS

def get_input_data(tf):
    noise = loadmat('/home/phil/matlab.mat')['noise'].T
    nc = noise.shape[1]
    spike_proto_sc = sp.cos(sp.linspace(-sp.pi, 3 * sp.pi, tf))
    spike_proto_sc *= sp.hanning(tf)
    scale = sp.linspace(0, 2, tf)
    cvals = [(5., .5), (4., 9.), (3., 3.), (7., 2.5)]
    xi1 = sp.vstack([spike_proto_sc * cvals[i][0] * scale
                     for i in xrange(nc)]).T
    xi2 = sp.vstack([spike_proto_sc * cvals[i][1] * scale[::-1]
                     for i in xrange(nc)]).T
    temps = sp.asarray([xi1, xi2])
    ce = TimeSeriesCovE.white_noise_init(tf, nc, std=.98)
    signal = sp.zeros_like(noise)
    NPOS = 4
    LEN = len(noise)
    POS = [(int(i * LEN / (NPOS + 1)), 100) for i in xrange(1, NPOS + 1)]
    POS.append((100, 2))
    POS.append((150, 2))
    print POS
    for pos, tau in POS:
        signal[pos:pos + tf] += temps[0]
        signal[pos + tau:pos + tau + tf] += temps[1]
    return signal, noise, ce, temps


def load_input_data(tf):
    MAT = loadmat('/home/phil/matlab.mat')
    noise = MAT['noise'].T
    signal = MAT['signal'].T
    nc = noise.shape[1]
    ce = TimeSeriesCovE.white_noise_init(tf, nc, std=.98)
    temps_ml = MAT['T']
    temps = sp.empty((temps_ml.shape[0], temps_ml.shape[1] / nc, nc))
    for i in xrange(temps_ml.shape[0]):
        temps[i] = mcvec_from_conc(temps_ml[i], nc=nc)
    return signal, noise, ce, temps


def test(debug=True):
    from spikeplot import plt, mcdata

    # setup
    TF = 47
    signal, noise, ce, temps = load_input_data(TF)
    FB = BOTMNode(templates=temps,
                  ce=ce,
                  adapt_templates=10,
                  learn_noise=False,
                  debug=debug,
                  ovlp_taus=None)
    x = sp.ascontiguousarray(signal + noise, dtype=sp.float32)

    # sort
    FB(x)
    print FB.rval

if __name__ == '__main__':
    test()
