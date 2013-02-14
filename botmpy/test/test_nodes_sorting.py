# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012 Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Neural Information Processing Group (NI)
#               School for Electrical Engineering and Computer Science
#               Berlin Institute of Technology
#               MAR 5-6, Marchstr. 23, 10587 Berlin, Germany
#               http://www.ni.tu-berlin.de/
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

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

import scipy as sp
from botmpy.common import TimeSeriesCovE, VERBOSE
from botmpy.nodes import BOTMNode
from numpy.testing import assert_array_almost_equal

##---TESTS

class TestSortingNodes(ut.TestCase):
    def setUp(self):
        pass

    def testMainSingle(self, verbose=VERBOSE.PLOT):
        import time

        # setup
        V = VERBOSE(verbose)
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
        FB = BOTMNode(
            templates=templates,
            ce=ce,
            verbose=V,
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

        # test against
        if V.has_print:
            print '### constructed spike times ###'
        test_u0 = sorted([t_tpl[0] for t_tpl in POS])
        test_u1 = sorted([t_tpl[0] + t_tpl[1] for t_tpl in POS])
        test_rval = {0: sp.array(test_u0) + TF / 2, 1: sp.array(test_u1) + TF / 2}
        if V.has_print:
            print test_rval

        # sort
        tic_o = time.clock()
        FB(x)
        toc_o = time.clock()
        if V.has_print:
            print '### sorting spike times ###'
            print FB.rval

        if V.has_plot:
            FB.plot_template_set(show=False)
            FB.plot_sorting(show=True)

        if V.has_print:
            print '###'
            print 'duration:', toc_o - tic_o

        for k in FB.rval:
            assert_array_almost_equal(FB.rval[k], test_rval[k], decimal=0)

if __name__ == '__main__':
    ut.main()
