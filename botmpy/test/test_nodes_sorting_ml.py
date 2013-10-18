# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012-2013, Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Philipp Meier <pmeier82@gmail.com>
#
#               Neural Information Processing Group (NI)
#               School for Electrical Engineering and Computer Science
#               Berlin Institute of Technology
#               MAR 5-6, Marchstr. 23, 10587 Berlin, Germany
#               http://www.ni.tu-berlin.de/
#
# Repository:   https://github.com/pmeier82/BOTMpy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal with the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimers.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimers in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of Neural Information Processing Group (NI), Berlin
#   Institute of Technology, nor the names of its contributors may be used to
#   endorse or promote products derived from this Software without specific
#   prior written permission.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# WITH THE SOFTWARE.
#_____________________________________________________________________________
#
# Acknowledgements:
#   Philipp Meier <pmeier82@gmail.com>
#_____________________________________________________________________________
#
# Changelog:
#   * <iso-date> <identity> :: <description>
#_____________________________________________________________________________
#

## IMPORTS

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from scipy.io import loadmat
from botmpy.common import TimeSeriesCovE, mcvec_from_conc, VERBOSE
from botmpy.nodes import BOTMNode
from spikeplot import plt

plt.interactive(False)

## TESTS

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
    FB = BOTMNode(
        templates=temps,
        ce=ce,
        adapt_templates=10,
        learn_noise=False,
        verbose=VERBOSE(debug * 10),
        ovlp_taus=None,
        chunk_size=500)
    x = sp.ascontiguousarray(signal + noise, dtype=sp.float32)

    # sort
    FB.plot_xvft()
    FB(x)
    FB.plot_sorting()
    print FB.rval
    plt.show()

if __name__ == '__main__':
    test()
