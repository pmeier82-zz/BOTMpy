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

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from botmpy.common import MxRingBuffer

##---TESTS

class TestMxRingbuffer(ut.TestCase):
    def setUp(self):
        self.rb = MxRingBuffer(6, (4, 4))

    def testInit(self):
        """test initial states"""

        self.assertEqual(self.rb.capacity, 6)
        self.assertEqual(len(self.rb), 0)
        self.assertTupleEqual(self.rb.dimension, (4, 4))

    def testInsertion(self):
        """test for append and extend"""

        self.rb.append(sp.eye(4))
        self.assertEqual(len(self.rb), 1)
        assert_equal(self.rb[0], sp.eye(4))

        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(1, 4)])
        self.assertEqual(len(self.rb), 4)
        assert_equal(self.rb[-1], sp.eye(4) * 4.0)

    def testFill(self):
        """test buffer filling"""

        self.rb.fill(sp.eye(4))
        self.assertEqual(len(self.rb), 6)
        for item in self.rb:
            assert_equal(item, sp.eye(4))

    def testCapacityIncrease(self):
        """test increase of capacity during operation"""

        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(6)])
        self.assertEqual(len(self.rb), 6)
        for i, item in enumerate(self.rb):
            assert_equal(item, sp.eye(4) * (i + 1))
        self.rb.capacity = 10
        self.assertEqual(len(self.rb), 6)
        for i, item in enumerate(self.rb):
            assert_equal(item, sp.eye(4) * (i + 1))

    def testCapacityDecrease(self):
        """test decrease of capacity during operation"""

        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(6)])
        self.assertEqual(len(self.rb), 6)
        for i, item in enumerate(self.rb):
            assert_equal(item, sp.eye(4) * (i + 1))
        self.rb.capacity = 4
        self.assertEqual(len(self.rb), 4)
        for i, item in enumerate(self.rb):
            assert_equal(item, sp.eye(4) * (i + 3))

    def testFlush(self):
        """tes for clear of ringbuffer"""

        self.rb.fill(sp.ones(self.rb.dimension))
        rb = self.rb.flush()
        assert_equal(rb, sp.array([
        sp.ones(self.rb.dimension) for _ in xrange(6)
        ]))
        self.assertEqual(len(self.rb), 0)

    def testMean(self):
        """mean calculation"""

        # empty
        assert_equal(self.rb.mean(), sp.zeros(self.rb.dimension))

        # ones filled
        self.rb.fill(sp.ones(self.rb.dimension))
        assert_equal(self.rb.mean(), sp.ones(self.rb.dimension))

        # eye xrange
        self.rb.clear()
        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(6)])
        assert_equal(self.rb.mean(), sp.eye(4) * 3.5)
        assert_equal(self.rb.mean(2), sp.eye(4) * 5.5)
        assert_equal(self.rb.mean(1), sp.eye(4) * 6.0)

    def testIndexing(self):
        """test for indexing elements and slices"""

        # when empty
        self.assertRaises(IndexError, self.rb.__getitem__, 0)

        # after fill to cap
        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(6)])
        for i in xrange(6):
            assert_equal(self.rb[i], sp.eye(4) * (i + 1))
        assert_equal(self.rb[:2], sp.array([sp.eye(4), sp.eye(4) * 2]))

        # after wrap around
        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(9)])
        for i in xrange(6):
            assert_equal(self.rb[i], sp.eye(4) * (i + 4))
        assert_equal(self.rb[:2], sp.array([sp.eye(4) * 4, sp.eye(4) * 5]))

if __name__ == '__main__':
    ut.main()
