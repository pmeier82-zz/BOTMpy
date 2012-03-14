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


"""implementation of mc filter application with time domain FIR filters

There is a general implementation suitable for larger chunks of data and batch
mode filtering `mcfilter`. Also there is a filtering strategy that implements
a history item for the part of the signal that could not be filtered in case
of chunked continuous data, so that the filter is applied as far as possible
and the relevant part of the signal data thas is needed to process the next
chunk is passed on as the so called history item.

Implementations are given in Python and alternatively as in Cython. On
import the Cython function is being tried to load, on failure the python
version is loaded as a fallback.
"""

"""multi-channeled filtering package"""
__docformat__ = 'restructuredtext'
__all__ = ['mcfilter', 'mcfilter_hist']

##---IMPORTS

import scipy as sp
from ..funcs_general import mcvec_from_conc
from .mcfilter_py import *
from .mcfilter_cy import *

##---CYTHON

VERBOSE = True

#try:
#    import mcfilter_cython
#
#    if VERBOSE is True:
#        print 'using cython!'
#
#except ImportError:
#    import mcfilter_py
#
#    if VERBOSE is True:
#        print 'using python!'

##---FUNCTIONS

def _mcfilter_check(mc_data, mc_filt):
    return mc_data, mc_filt


def _mcfilter_hist_check(mc_data, mc_filt, mc_hist):
    dtype = mc_data.dtype
    if dtype not in [sp.float32, sp.float64]:
        dtype = sp.float32
    mc_data = sp.ascontiguousarray(mc_data, dtype=dtype)
    mc_filt = sp.ascontiguousarray(mc_filt, dtype=dtype)
    if mc_hist is None:
        mc_hist = sp.zeros(
            (mc_filt.shape[0] - 1,
             mc_filt.shape[1]), dtype=dtype)
    mc_hist = sp.ascontiguousarray(mc_data, dtype=dtype)
    return mc_data, mc_filt, mc_hist


def mcfilter(mc_data, mc_filt):
    """filter a multichanneled signal with a multichanneled filter

    This is the python implementation for batch mode filtering.

    We do not need to account for zero padding, as we are only interested in
    the 'same' size vector of the xcorr.

    :Parameters:
        mc_data : ndarray
            Data for one channel per columsp.
        mc_filt : ndarray
            A multichanneled finite impulse response filter with either:
            channels concatenate or the filter for each channel on one column.
        correlate_mode : str
            string to pass to scipy.correlate
            Default='same'
    :Returns:
        ndarray
            filtered signal (same shape as data)
    """

    # checks and inits
    if mc_data.ndim != mc_filt.ndim > 2:
        raise ValueError('wrong dimensions: %s, %s' %
                         (mc_data.shape, mc_filt.shape))
    if mc_data.ndim == 1:
        mc_data = sp.atleast_2d(mc_data).T
    nc = mc_data.shape[1]
    if mc_filt.ndim == 1:
        mc_filt = mcvec_from_conc(mc_filt, nc=nc)
    if mc_data.shape[1] != mc_filt.shape[1]:
        raise ValueError('channel count does not match')

    # filter the signal
    return sp.sum(
        [sp.correlate(mc_data[:, c], mc_filt[:, c], mode=correlate_mode)
         for c in xrange(nc)], axis=0)


def mcfilter_hist(mc_data, mc_filt, mc_hist=None):
    """filter a multichanneled signal with a multichanneled filter

    We dont need to account for zero padding, as we are only interested in the
    'same' size vector of the xcorr.

    :Parameters:
        mc_data : ndarray
            Data for one channel per columsp.
        mc_filt : ndarray
            A multichanneled finite impulse response filter with either:
            channels concatenate or the filter for each channel on one column.
        hist_item : ndarray
            data history to prepend to the data for filter
    :Returns:
        ndarray
            filtered signal (same shape as data)
        ndarray
            new history item for next filter step
    """

    # checks and inits
    if mc_data.ndim != mc_filt.ndim > 2:
        raise ValueError('wrong dimensions: %s, %s' %
                         (mc_data.shape, mc_filt.shape))
    if mc_data.ndim == 1:
        mc_data = sp.atleast_2d(mc_data).T
    td, nc = mc_data.shape
    if mc_filt.ndim == 1:
        mc_filt = mcvec_from_conc(mc_filt, nc=nc)
    if mc_data.shape[1] != mc_filt.shape[1]:
        raise ValueError('channel count does not match')
    tf = mc_filt.shape[0]
    if mc_hist is None:
        mc_hist = sp.zeros((tf - 1, nc))
    th = mc_hist.shape[0]
    if th + 1 != tf:
        raise ValueError(
            'len(history)+1[%d] != len(filter)[%d]' % (th + 1, tf))
    mc_data = sp.vstack((mc_hist, mc_data))
    rval = sp.zeros(td, dtype=mc_data.dtype)

    # filter the signal (by hand)
    for t in xrange(td):
        for c in xrange(nc):
            rval[t] += sp.dot(mc_data[t:t + tf, c], mc_filt[:, c])

    # return
    return rval, mc_data[t + 1:, :].copy()

#if mcfilter_cython is not None:
#    def _mcfilter_hist_cy(mc_data, mc_filt, mc_hist=None):
#        dtype = mc_data.dtype
#        if dtype not in [sp.float32, sp.float64]:
#            dtype = sp.float32
#        mc_data = sp.ascontiguousarray(mc_data, dtype=dtype)
#        mc_filt = sp.ascontiguousarray(mc_filt, dtype=dtype)
#        if mc_hist is None:
#            mc_hist = sp.zeros(
#                (mc_filt.shape[0] - 1,
#                 mc_filt.shape[1]), dtype=dtype)
#        mc_hist = sp.ascontiguousarray(mc_data, dtype=dtype)
#        if dtype == sp.float32:
#            return mcfilter_cython._mcfilter_hist_cy32(
#                mc_data, mc_filt, mc_hist)
#        else:
#            return mcfilter_cython._mcfilter_hist_cy64(
#                mc_data, mc_filt, mc_hist)
#mcfilter_hist = _mcfilter_hist_cy or _mcfilter_hist_py
#mcfilter_hist.__doc__ = str("multichanneled filter application without "
#                            "causal part of the filter")

if __name__ == '__main__':
    pass
