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

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

import scipy as sp
from numpy.testing import assert_equal, assert_almost_equal

from botmpy.mcfilter._cy_mcfilter import mcfilter_hist_f32, mcfilter_hist_f64
from botmpy.mcfilter._py_mcfilter import mcfilter, mcfilter_hist

## TESTS

class TestMcFilterHist(ut.TestCase):
    def testHistoryPy(self):
        """test history and filter accuracy, python"""

        ## init
        tf = 45
        nc = 4
        ns = 10000
        dt = sp.float64
        data = sp.randn(ns, nc).astype(dt)
        mean_at_ns_halve = data[ns / 2:ns / 2 + tf].mean()
        hist = sp.zeros((tf - 1, nc), dtype=dt)
        # rolling average filter
        filt = sp.ones((tf, nc), dtype=dt) / (tf * nc)
        fout, hist = mcfilter_hist(data, filt, hist)

        ## test
        assert_equal(hist, data[-(tf - 1):])
        assert_almost_equal(fout[ns / 2 + tf - 1], mean_at_ns_halve, decimal=15)

    def testHistoryCy32(self):
        """test history and filter accuracy, cython float32"""

        ## init
        tf = 45
        nc = 4
        ns = 10000
        dt = sp.float32
        data = sp.randn(ns, nc).astype(dt)
        mean_at_ns_halve = data[ns / 2:ns / 2 + tf].mean()
        hist = sp.zeros((tf - 1, nc), dtype=dt)
        # rolling average filter
        filt = sp.ones((tf, nc), dtype=dt) / (tf * nc)
        fout, hist = mcfilter_hist_f32(data, filt, hist)

        ## test
        assert_equal(hist, data[-(tf - 1):])
        assert_almost_equal(fout[ns / 2 + tf - 1], mean_at_ns_halve, decimal=6)

    def testHistoryCy64(self):
        """test history and filter accuracy, cython float64"""

        ## init
        tf = 45
        nc = 4
        ns = 10000
        dt = sp.float64
        data = sp.randn(ns, nc).astype(dt)
        mean_at_ns_halve = data[ns / 2:ns / 2 + tf].mean()
        hist = sp.zeros((tf - 1, nc), dtype=dt)
        # rolling average filter
        filt = sp.ones((tf, nc), dtype=dt) / (tf * nc)
        fout, hist = mcfilter_hist_f64(data, filt, hist)

        ## test
        assert_equal(hist, data[-(tf - 1):])
        assert_almost_equal(fout[ns / 2 + tf - 1], mean_at_ns_halve, decimal=15)

    def testPyVsCyOnesCy32(self):
        """test python and cython equivalent (using ones), float32"""

        tf = 3
        nc = 2
        ns = 10000
        dt = sp.float32
        data = sp.ones((ns, nc), dtype=dt)
        filt = sp.ones((tf, nc), dtype=dt)
        hist_py = sp.ones((tf - 1, nc), dtype=dt)
        hist_cy = sp.ones((tf - 1, nc), dtype=dt)
        fopy, hopy = mcfilter_hist(data, filt, hist_py)
        focy, hocy = mcfilter_hist_f32(data, filt, hist_cy)
        assert_almost_equal(fopy, focy, decimal=6)
        assert_equal(fopy[sp.random.randint(ns)], tf * nc)
        assert_equal(focy[sp.random.randint(ns)], tf * nc)

    def testPyVsCyOnesCy64(self):
        """test python and cython equivalent (using ones), float64"""

        tf = 3
        nc = 2
        ns = 10000
        dt = sp.float64
        data = sp.ones((ns, nc), dtype=dt)
        filt = sp.ones((tf, nc), dtype=dt)
        hist_py = sp.ones((tf - 1, nc), dtype=dt)
        hist_cy = sp.ones((tf - 1, nc), dtype=dt)
        fopy, hopy = mcfilter_hist(data, filt, hist_py)
        focy, hocy = mcfilter_hist_f64(data, filt, hist_cy)
        assert_almost_equal(fopy, focy, decimal=15)
        assert_equal(fopy[sp.random.randint(ns)], tf * nc)
        assert_equal(focy[sp.random.randint(ns)], tf * nc)

    def testPyVsCyZerosCy32(self):
        """test python and cython equivalent (using zeros), float32"""

        tf = 3
        nc = 2
        ns = 10000
        dt = sp.float32
        data = sp.randn(ns, nc).astype(dt)
        filt = sp.zeros((tf, nc), dtype=dt)
        hist_py = sp.zeros((tf - 1, nc), dtype=dt)
        hist_cy = sp.zeros((tf - 1, nc), dtype=dt)
        fopy, hopy = mcfilter_hist(data, filt, hist_py)
        focy, hocy = mcfilter_hist_f32(data, filt, hist_cy)
        assert_almost_equal(fopy, focy, decimal=6)
        assert_equal(fopy[sp.random.randint(ns)], 0.0)
        assert_equal(focy[sp.random.randint(ns)], 0.0)

    def testPyVsCyZerosCy64(self):
        """test python and cython equivalent (using zeros), float64"""

        tf = 3
        nc = 2
        ns = 10000
        dt = sp.float64
        data = sp.randn(ns, nc).astype(dt)
        filt = sp.zeros((tf, nc), dtype=dt)
        hist_py = sp.zeros((tf - 1, nc), dtype=dt)
        hist_cy = sp.zeros((tf - 1, nc), dtype=dt)
        fopy, hopy = mcfilter_hist(data, filt, hist_py)
        focy, hocy = mcfilter_hist_f64(data, filt, hist_cy)
        assert_almost_equal(fopy, focy, decimal=15)
        assert_equal(fopy[sp.random.randint(ns)], 0.0)
        assert_equal(focy[sp.random.randint(ns)], 0.0)

    def testPyVsCyRandnCy32(self):
        """test python and cython equivalent (using random numbers), float32"""

        tf = 3
        nc = 2
        ns = 10000
        dt = sp.float32
        data = sp.randn(ns, nc).astype(dt)
        filt = sp.zeros((tf, nc), dtype=dt)
        hist_py = sp.zeros((tf - 1, nc), dtype=dt)
        hist_cy = sp.zeros((tf - 1, nc), dtype=dt)
        fopy, hopy = mcfilter_hist(data, filt, hist_py)
        focy, hocy = mcfilter_hist_f32(data, filt, hist_cy)
        assert_almost_equal(fopy, focy, decimal=6)

    def testPyVsCyRandnCy64(self):
        """test python and cython"""

        tf = 3
        nc = 2
        ns = 10000
        dt = sp.float64
        data = sp.randn(ns, nc).astype(dt)
        filt = sp.zeros((tf, nc), dtype=dt)
        hist_py = sp.zeros((tf - 1, nc), dtype=dt)
        hist_cy = sp.zeros((tf - 1, nc), dtype=dt)
        fopy, hopy = mcfilter_hist(data, filt, hist_py)
        focy, hocy = mcfilter_hist_f64(data, filt, hist_cy)
        assert_almost_equal(fopy, focy, decimal=15)

    def testDataConcatenationPy(self):
        """history concatenation, python"""

        tf = 5
        middle = sp.floor(tf / 2.)
        nc = 1
        ns = 10000
        dt = sp.float64
        data = sp.zeros((ns, nc), dtype=dt)
        data[sp.arange(0, ns, 10)] = 1.0
        filt = sp.zeros((tf, nc), dtype=dt)
        filt[middle] = 1.0
        hist = sp.zeros((tf - 1, nc), dtype=dt)
        fout, hout = mcfilter_hist(data, filt, hist)
        assert_equal(data[:-middle], sp.array([fout[middle:]]).T)

    def testDataConcatenationCy32(self):
        """history concatenation, float32"""

        tf = 5
        middle = sp.floor(tf / 2.)
        nc = 1
        ns = 10000
        dt = sp.float32
        data = sp.zeros((ns, nc), dtype=dt)
        data[sp.arange(0, ns, 10)] = 1.0
        filt = sp.zeros((tf, nc), dtype=dt)
        filt[middle] = 1.0
        hist = sp.zeros((tf - 1, nc), dtype=dt)
        fout, hout = mcfilter_hist_f32(data, filt, hist)
        assert_equal(data[:-middle], sp.array([fout[middle:]]).T)

    def testDataConcatenationCy64(self):
        """history concatenation, float64"""

        tf = 5
        middle = sp.floor(tf / 2.)
        nc = 1
        ns = 10000
        dt = sp.float32
        data = sp.zeros((ns, nc), dtype=dt)
        data[sp.arange(0, ns, 10)] = 1.0
        filt = sp.zeros((tf, nc), dtype=dt)
        filt[middle] = 1.0
        hist = sp.zeros((tf - 1, nc), dtype=dt)
        fout, hout = mcfilter_hist_f32(data, filt, hist)
        assert_equal(data[:-middle], sp.array([fout[middle:]]).T)

## MAIN

if __name__ == "__main__":
    ut.main()

## EOF
