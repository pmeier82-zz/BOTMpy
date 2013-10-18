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

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

import scipy as sp

from numpy.testing import assert_equal

from botmpy.common import (TimeSeriesCovE, mcvec_to_conc,
                            mcvec_from_conc)
from botmpy.nodes import MatchedFilterNode, NormalisedMatchedFilterNode


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

    def testFilterTrivial(self):
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
