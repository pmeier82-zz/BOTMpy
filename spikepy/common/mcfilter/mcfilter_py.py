# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (C) 2011 by Philipp Meier, Felix Franke and
# Berlin Institute of Technology
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#_____________________________________________________________________________
#
# Affiliation:
#   Bernstein Center for Computational Neuroscience (BCCN) Berlin
#     and
#   Neural Information Processing Group
#   School for Electrical Engineering and Computer Science
#   Berlin Institute of Technology
#   FR 2-1, Franklinstrasse 28/29, 10587 Berlin, Germany
#   Tel: +49-30-314 26756
#_____________________________________________________________________________
#
# Acknowledgements:
#   This work was supported by Deutsche Forschungs Gemeinschaft (DFG) with
#   grant GRK 1589/1
#     and
#   Bundesministerium für Bildung und Forschung (BMBF) with grants 01GQ0743
#   and 01GQ0410.
#_____________________________________________________________________________
#

"""multichanneled filter application for time domain FIR filters

PYTHON IMPLEMENTATIONS USING SCIPY
"""
__docformat__ = 'restructuredtext'
__all__ = ['_mcfilter_py', '_mcfilter_hist_py', ]

##---IMPORTS

import scipy as sp

##---FUNCTIONS

def _mcfilter_py(mc_data, mc_filt):
    if mc_data.ndim != mc_filt.ndim > 2:
        raise ValueError('wrong dimensions: %s, %s' %
                         (mc_data.shape, mc_filt.shape))
    if mc_data.shape[1] != mc_filt.shape[1]:
        raise ValueError('channel count does not match')
    return sp.sum(
        [sp.correlate(mc_data[:, c], mc_filt[:, c], mode='same')
         for c in xrange(mc_data.shape[1])], axis=0)


def _mcfilter_hist_py(mc_data, mc_filt, mc_hist):
    if mc_data.ndim != mc_filt.ndim > 2:
        raise ValueError('wrong dimensions: %s, %s' %
                         (mc_data.shape, mc_filt.shape))
    if mc_data.shape[1] != mc_filt.shape[1]:
        raise ValueError('channel count does not match')
    mc_hist_and_data = sp.vstack((mc_hist, mc_data))
    rval = sp.zeros(mc_data.shape[0], dtype=mc_data.dtype)
    for t in xrange(mc_data.shape[0]):
        for c in xrange(mc_hist_and_data.shape[1]):
            rval[t] += sp.dot(mc_hist_and_data[t:t + mc_filt.shape[0], c],
                              mc_filt[:, c])
    return rval, mc_data[t + 1:, :].copy()

if __name__ == '__main__':
    pass
