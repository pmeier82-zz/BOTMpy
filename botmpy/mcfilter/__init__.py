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
__docformat__ = "restructuredtext"
__all__ = ["mcfilter", "mcfilter_hist", "CYTHON"]

## IMPORTS

import scipy as sp
import warnings

# suppress warnings after first presentation (most likely on import)
warnings.simplefilter('once')

import _py_mcfilter

try:
    import _cy_mcfilter

    CYTHON = True
except ImportError, ex:
    warnings.warn("Cython implementation not found! Falling back to Python!", ImportWarning)
    CYTHON = False

## FUNCTIONS

def mcfilter(mc_data, mc_filt):
    """filter a multi-channeled signal with a multi-channeled filter

    The signal will be padded with zeros on both ends to overcome filter
    artifacts. Input and output will have same dimensions.

    .. note:
        If you compiled the cython extension, this will use a faster c code
        implementation.

    :param ndarray mc_data: signal data [data_samples, channels]
    :param ndarray mc_filt: FIR filter [filter_samples, channels]
    :return: ndarray -- filtered signal [data_samples]
    """

    if mc_data.ndim != 2 or mc_filt.ndim != 2:
        raise ValueError("wrong dimensions: %s, %s" % (mc_data.shape, mc_filt.shape))
    if mc_data.shape[1] != mc_filt.shape[1]:
        raise ValueError("channel count does not match")

    if CYTHON is True:
        dtype = mc_data.dtype
        if dtype not in [sp.float32, sp.float64]:
            dtype = sp.float32
        mc_data, mc_filt = (
            sp.ascontiguousarray(mc_data, dtype=dtype),
            sp.ascontiguousarray(mc_filt, dtype=dtype))
        if dtype == sp.float32:
            return _cy_mcfilter.mcfilter_f32(mc_data, mc_filt)
        elif dtype == sp.float64:
            return _cy_mcfilter.mcfilter_f64(mc_data, mc_filt)
    else:
        return _py_mcfilter.mcfilter(mc_data, mc_filt)


def mcfilter_hist(mc_data, mc_filt, mc_hist=None):
    """filter a multi-channeled signal with a multi-channeled filter

    A history item will be used and prepended before applying the filter. This
    means, the last sample in the filter output is calculated for the case
    where the last samples of the data filter meet. The history form the
    current step is then returned and can be used for the next step. The
    history is Tf-1 samples long.

    :param ndarray mc_data: signal data [data_samples, channels]
    :param ndarray mc_filt: FIR filter [filter_samples, channels]
    :param ndarray mc_hist: history [hist_samples, channels]. the history is
        of size ´filter_samples - 1´. If None, this will be substituted with
        all zeros.
    :return: tuple(ndarray,ndarray) -- filter output [data_samples], history
        item [hist_samples, channels]
    """

    if mc_data.ndim != 2 or mc_filt.ndim != 2:
        raise ValueError("wrong dimensions: %s, %s" % (mc_data.shape, mc_filt.shape))
    if mc_data.shape[1] != mc_filt.shape[1]:
        raise ValueError("channel count does not match")
    if mc_hist is None:
        mc_hist = sp.zeros((mc_filt.shape[0] - 1, mc_data.shape[0]))
    if mc_hist.shape[0] + 1 != mc_filt.shape[0]:
        raise ValueError(
            "len(history)+1[%d] != len(filter)[%d]" %
                         ( mc_hist.shape[0] + 1, mc_filt.shape[0]))
    if CYTHON is True:
        dtype = mc_data.dtype
        if dtype not in [sp.float32, sp.float64]:
            dtype = sp.float32
        if mc_data.shape[1] != mc_filt.shape[1]:
            raise ValueError("channel count does not match")
        mc_data, mc_filt, mc_hist = (
            sp.ascontiguousarray(mc_data, dtype=dtype),
            sp.ascontiguousarray(mc_filt, dtype=dtype),
            sp.ascontiguousarray(mc_hist, dtype=dtype))
        try:
            return {sp.float32: _mcfilter_hist_cy32,
                    sp.float64: _mcfilter_hist_cy64}[dtype](mc_data, mc_filt, mc_hist)
        except:
            raise TypeError("dtype != float32 or float64: %s" % dtype)
    else:
        if mc_data.ndim != mc_filt.ndim > 2:
            raise ValueError("wrong dimensions: %s, %s" %
                             (mc_data.shape, mc_filt.shape))
        if mc_data.shape[1] != mc_filt.shape[1]:
            raise ValueError("channel count does not match")
        mc_hist_and_data = sp.vstack((mc_hist, mc_data))
        rval = sp.zeros(mc_data.shape[0], dtype=mc_data.dtype)
        for t in xrange(mc_data.shape[0]):
            for c in xrange(mc_hist_and_data.shape[1]):
                rval[t] += sp.dot(mc_hist_and_data[t:t + mc_filt.shape[0], c], mc_filt[:, c])
        return rval, mc_data[-(mc_hist.shape[0]):].copy()

## MAIN

if __name__ == "__main__":
    pass

## EOF
