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

from numpy.testing import assert_equal, assert_almost_equal
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

    def testClusteringKMeans(self):
        # print
        # print 'starting to cluster..',
        cls = HomoscedasticClusteringNode(clus_type='kmeans',
                                          crange=range(1, 15),
                                          debug=False)
        cls(self.data)
        labls = cls.labels
        means = sp.array([self.data[labls == i].mean(axis=0)
                          for i in xrange(labls.max() + 1)])
        # print 'done!'
        # print

        n_labls = labls.max() + 1
        simi_mx = sp.zeros((len(self.prio), n_labls))
        idx = sp.concatenate(([0], self.prio.cumsum()))
        for i in xrange(len(self.prio)):
            for j in xrange(n_labls):
                simi_mx[i, j] = (labls[idx[i]:idx[i + 1]] == j).sum()
        simi_map = simi_mx.argmax(axis=1)
        # print simi_mx
        # print simi_map
        # print
        labls_maped = sp.zeros_like(labls)
        for new, old in enumerate(simi_map):
            labls_maped[labls == old] = new

        # print
        # print 'labels found', labls_maped
        # print 'labels expected', self.labels
        # assert_almost_equal(labls_maped, self.labels, decimal=0,
        #                     err_msg='labels dont match')
        # print
        # print 'means found:', means[simi_map]
        # print 'means expected:', self.means
        assert_almost_equal(means[simi_map], self.means, decimal=0,
                            err_msg='means dont match')
        # print

        correct = (labls_maped == self.labels).sum() / float(self.labels.size)
        self.assertGreaterEqual(correct, 0.9)

        # print 'classification error: %s%%' % ((1.0 - correct) * 100.0)
        # print

    def testClusteringGMM(self):
        # print
        # print 'starting to cluster..',
        cls = HomoscedasticClusteringNode(clus_type='gmm',
                                          crange=range(1, 15),
                                          debug=False)
        cls(self.data)
        labls = cls.labels
        means = sp.array([self.data[labls == i].mean(axis=0)
                          for i in xrange(labls.max() + 1)])
        # print 'done!'
        # print

        n_labls = labls.max() + 1
        simi_mx = sp.zeros((len(self.prio), n_labls))
        idx = sp.concatenate(([0], self.prio.cumsum()))
        for i in xrange(len(self.prio)):
            for j in xrange(n_labls):
                simi_mx[i, j] = (labls[idx[i]:idx[i + 1]] == j).sum()
        simi_map = simi_mx.argmax(axis=1)
        # print simi_mx
        # print simi_map
        # print
        labls_maped = sp.zeros_like(labls)
        for new, old in enumerate(simi_map):
            labls_maped[labls == old] = new

        # print
        # print 'labels found', labls_maped
        # print 'labels expected', self.labels
        # assert_almost_equal(labls_maped, self.labels, decimal=0,
        #                     err_msg='labels dont match')


        # print
        # print 'means found:', means[simi_map]
        # print 'means expected:', self.means
        assert_almost_equal(means[simi_map], self.means, decimal=0,
                            err_msg='means dont match')
        # print

        correct = (labls_maped == self.labels).sum() / float(self.labels.size)
        self.assertGreaterEqual(correct, 0.9)
        # print 'classification error: %s%%' % ((1.0 - correct) * 100# .0)
        # print
        cls.plot(self.data, show=True)

    """

    # paramters for HomoscedasticClusteringNode
    # clus_type = 'kmeans', crange = range(1, 16), maxiter = 32,
    # repeats = 4, conv_th = 1e-4, sigma_factor = 4.0,
    # uprior = False, dtype = sp.float32, debug = False):

    mul = 2.0
    dim = 6
    data = sp.vstack(
        [sp.randn(50 * (i + 1), dim) + [5 * i * (-1) ** i] * dim for
         i in
         xrange(5)]) * mul
    HCN = HomoscedasticClusteringNode(clus_type='gmm',
                                      crange=[1, 2, 3, 4, 5, 6, 7,
                                      8, 9],
                                      maxiter=128,
                                      repeats=4,
                                      sigma_factor=mul * mul,
                                      weights_uniform=False,
                                      debug=True)
    HCN(data)
    HCN.plot(data, views=3, show=True)
    """

if __name__ == '__main__':
    ut.main()
