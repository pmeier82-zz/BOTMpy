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

"""detector nodes using the nonlinear energy operator"""
__docformat__ = 'restructuredtext'
__all__ = ['SDKteoNode', 'SDMneoNode']

## IMPORTS

import scipy as sp
from scipy.stats.mstats import mquantiles
from botmpy.common import k_neo, m_neo
from .threshold_detector import ThresholdDetectorNode

## CLASSES

class SDKteoNode(ThresholdDetectorNode):
    """spike detector node

    energy: nonlinear energy operator
    threshold: qualtile
    """

    def __init__(self, k_value=1, quantile=0.98, **kwargs):
        """
        :Parameters:
            see ThresholdDetectorNode

            kvalue : int
                Integer determining the k_neo detector resolution.
        """

        # super
        kwargs.update(
            threshold_base='energy',
            threshold_factor=kwargs.get('threshold_factor', 0.96),
            min_dist=kwargs.get('min_dist', 5),
            ch_separate=True)
        super(SDKteoNode, self).__init__(**kwargs)

        # members
        self.k_value = int(k_value)
        self.quantile = quantile

    def _energy_func(self, x):
        return sp.vstack([k_neo(x[:, c], k=self.k_value)
                          for c in xrange(x.shape[1])]).T

    def _threshold_func(self, x):
        return mquantiles(x, prob=[self.quantile])[0]


class SDMneoNode(ThresholdDetectorNode):
    """spike detector node

    energy: multi-resolution nonlinear energy operator
    threshold: qualtile
    """

    def __init__(self, k_values=None, quantile=0.98, **kwargs):
        """uses the multi-resolution nonlinear energy operator to detect spikes

        The m_neo will be called with `reduce`=True.

        :param list k_values: k-values for the k_neo instances.
        :param list k_values: list of int determining the k_neo detectors to
            build for the m_neo neo detector from.
        :param float quantile: threshold as a quantile of the reduced m_neo output
        """

        # super
        kwargs.update(
            threshold_base='energy',
            threshold_factor=kwargs.get('threshold_factor', 0.96),
            min_dist=kwargs.get('min_dist', 5),
            ch_separate=True)
        super(SDMneoNode, self).__init__(**kwargs)

        # members
        self.k_values = map(int, k_values or [1, 3, 5, 7, 9])
        self.quantile = quantile

    def _energy_func(self, x):
        return sp.vstack([m_neo(x[:, c], k_values=self.k_values, reduce=True)
                          for c in xrange(x.shape[1])]).T

    def _threshold_func(self, x):
        return mquantiles(x, prob=[self.quantile])[0]

## MAIN

if __name__ == '__main__':
    pass

## EOF
