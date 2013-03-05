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
from botmpy.nodes.spike_detection import *
from numpy.testing import assert_array_almost_equal
from spikeplot import mcdata

##---TESTS

class TestSortingNodes(ut.TestCase):
    def setUp(self):
        self.bkg = sp.randn(100, 2)
        self.data = sp.zeros_like(self.bkg)

        ev_kernel = sp.sin(sp.linspace(0, sp.pi, 7))
        self.events = []
        self.events.append(20)#, 50, 70, 80
        self.data[17:24] += sp.vstack((ev_kernel * 5, ev_kernel * 10)).T

        self.data += self.bkg
        #mcdata(self.data)

    def testThresholdDetetorBase(self, verbose=VERBOSE.PLOT):
        SD = ThresholdDetectorNode()
        self.assertIsNone(SD.events)
        self.assertEqual(SD.data, [])
        SD(self.data)
        self.assertIsNotNone(SD.events)
        assert_array_almost_equal(SD.data, self.data)

    # TODO: interface tests

    def testSDSqr(self):
        SD = SDSqrNode()
        SD(self.data)
        print SD.events
        print SD.threshold

if __name__ == '__main__':
    ut.main()
