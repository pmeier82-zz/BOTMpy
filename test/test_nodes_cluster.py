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

from numpy.testing import assert_equal, assert_array_almost_equal
import scipy as sp
from botmpy.nodes import HomoscedasticClusteringNode

## TESTS

class TestClusterNodes(ut.TestCase):
    def setUp(self):
        self.ndim = sp.array([50, 40, 20, 5])
        self.prio = sp.array([50, 100, 150, 20, 350,
                              50, 100, 150, 20, 350])
        self.ncls = len(self.prio)
        self.means = sp.array([sp.random.random_sample(len(self.ndim))
                               for _ in xrange(len(self.prio))])
        self.means = self.means * 2 * self.ndim - self.ndim
        self.data = sp.randn(sum(self.prio), len(self.ndim)) * 2
        self.labels = sp.zeros((sum(self.prio)))
        idx = 0
        for c, n in enumerate(self.prio):
            self.data[idx:idx + n] += self.means[c]
            self.labels[idx:idx + n] = c
            idx += n

    def _clusterRun(self, mode):

        ## init
        cls = HomoscedasticClusteringNode(
            clus_type=mode,
            crange=range(1, 15),
            debug=False,
            alpha=1e-2)
        cls(self.data)
        labls = cls.labels
        means = sp.array(
            [self.data[labls == i].mean(axis=0)
             for i in xrange(labls.max() + 1)])

        ## test stats
        n_labls = labls.max() + 1
        simi_mx = sp.zeros((len(self.prio), n_labls))
        idx = sp.concatenate(([0], self.prio.cumsum()))
        for i in xrange(len(self.prio)):
            for j in xrange(n_labls):
                simi_mx[i, j] = (labls[idx[i]:idx[i + 1]] == j).sum()
        simi_map = simi_mx.argmax(axis=1) # responsibilities
        labls_maped = sp.zeros_like(labls)
        for new, old in enumerate(simi_map):
            labls_maped[labls == old] = new

        # tests
        assert_array_almost_equal(
            means[simi_map], self.means, decimal=-1,
            err_msg='means do not match!')
        correct = (labls_maped == self.labels).sum() / float(self.labels.size)
        self.assertGreaterEqual(correct, 0.9)

    def testClusteringKmeans(self):
        """using the 'kmeans' mode"""

        self._clusterRun('kmeans')

    def testClusteringGmm(self):
        """using the 'gmm' mode"""

        self._clusterRun('gmm')

    def testClusteringDpGmm(self):
        """using the 'dpgmm' mode"""

        # self._clusterRun('dpgmm')

    def testClusteringMeanShift(self):
        """using the 'meanshift' mode"""

        # self._clusterRun('meanshift')

## MAIN

if __name__ == "__main__":
    ut.main()

## EOF
