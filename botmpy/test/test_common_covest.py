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

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from botmpy.common import TimeSeriesCovE

##---TESTS

class TestCovarianceEstimator(ut.TestCase):
    def setUp(self):
        self.dlen = 250000
        self.tf = 65
        self.nc = 4
        self.white_noise = sp.randn(self.dlen, self.nc)
        self.CE = TimeSeriesCovE(tf_max=self.tf, nc=self.nc)
        self.CE.new_chan_set((1, 2))
        self.CE.update(self.white_noise)

    def testTrivial(self):
        p_4_20 = {'tf':20, 'chan_set':(0, 1, 2, 3)}
        C_4_20 = self.CE.get_cmx(**p_4_20)
        self.assertTupleEqual(C_4_20.shape, (4 * 20, 4 * 20 ))
        assert_equal(C_4_20, C_4_20.T)

        p_2_10 = {'tf':10, 'chan_set':(0, 1)}
        C_2_10 = self.CE.get_cmx(**p_2_10)
        self.assertTupleEqual(C_2_10.shape, (2 * 10, 2 * 10 ))
        assert_equal(C_2_10, C_2_10.T)

    def testInverse(self):
        p_4_20 = {'tf':20, 'chan_set':(0, 1, 2, 3)}
        C_4_20 = self.CE.get_cmx(**p_4_20)
        iC_4_20 = self.CE.get_icmx(**p_4_20)
        should_be_eye80 = sp.dot(C_4_20, iC_4_20)
        assert_almost_equal(should_be_eye80, sp.eye(80), decimal=5)

        p_2_10 = {'tf':10, 'chan_set':(0, 1)}
        C_2_10 = self.CE.get_cmx(**p_2_10)
        iC_2_10 = self.CE.get_icmx(**p_2_10)
        should_be_eye20 = sp.dot(C_2_10, iC_2_10)
        assert_almost_equal(should_be_eye20, sp.eye(20), decimal=5)

##---MAIN

if __name__ == '__main__':
    ut.main()
