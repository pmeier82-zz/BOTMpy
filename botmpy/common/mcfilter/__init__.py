# -*- coding: utf-8 -*-
# _____________________________________________________________________________
#
# Copyright (c) 2012 Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Philipp Meier <pmeier82@gmail.com>
# Neural Information Processing Group (NI)
# School for Electrical Engineering and Computer Science
# Berlin Institute of Technology
# MAR 5-6, Marchstr. 23, 10587 Berlin, Germany
# http://www.ni.tu-berlin.de/
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
__all__ = ["mcfilter", "mcfilter_hist", "CYTHON_AVAILABLE"]

## IMPORTS

import scipy as sp
import warnings

warnings.simplefilter("once")

## USE_CYTHON

try:
    from .mcfilter_cy import (_mcfilter_cy32, _mcfilter_cy64, _mcfilter_hist_cy32, _mcfilter_hist_cy64)

    CYTHON_AVAILABLE = True
except ImportError, ex:
    from .mcfilter_py import _mcfilter_py, _mcfilter_hist_py

    warnings.warn("Cython implementation not found! Falling back to Python!\n{}".format(ex), ImportWarning)
    CYTHON_AVAILABLE = False

##---FUNCTIONS

def mcfilter(mc_data, mc_filt):
    """filter a multi-channeled signal with a multi-channeled filter

    This is the Python implementation for batch mode filtering. The signal
    will be zeros on both ends to overcome filter artifacts.

    :type mc_data: ndarray
    :param mc_data: signal data [data_samples, channels]
    :type mc_filt: ndarray
    :param mc_filt: FIR filter [filter_samples, channels]
    :rtype: ndarray
    :returns: filtered signal [data_samples]
    """

    if CYTHON_AVAILABLE is True:
        dtype = mc_data.dtype
        if dtype not in [sp.float32, sp.float64]:
            dtype = sp.float32
        if mc_data.shape[1] != mc_filt.shape[1]:
            raise ValueError("channel count does not match")
        mc_data, mc_filt = (sp.ascontiguousarray(mc_data, dtype=dtype),
                            sp.ascontiguousarray(mc_filt, dtype=dtype))
        if dtype == sp.float32:
            return _mcfilter_cy32(mc_data, mc_filt)
        elif dtype == sp.float64:
            return _mcfilter_cy64(mc_data, mc_filt)
        else:
            raise TypeError("dtype is not float32 or float64: %s" % dtype)
    else:
        return _mcfilter_py(mc_data, mc_filt)


def mcfilter_hist(mc_data, mc_filt, mc_hist=None):
    """filter a multichanneled signal with a multichanneled fir filter

    This is the Python implementation for online mode filtering with a
    chunk-wise history item, holding the last samples of tha preceding chunk.

    :type mc_data: ndarray
    :param mc_data: signal data [data_samples, channels]
    :type mc_filt: ndarray
    :param mc_filt: FIR filter [filter_samples, channels]
    :type mc_hist:
    :param mc_hist: history [hist_samples, channels]. the history is of size
        ´filter_samples - 1´. If None, this will be substituted with zeros.
    :rtype: tuple(ndarray,ndarray)
    :returns: filter output [data_samples], history item [hist_samples,
        channels]
    """

    if mc_hist is None:
        mc_hist = sp.zeros((mc_filt.shape[0] - 1, mc_data.shape[0]))
    if mc_hist.shape[0] + 1 != mc_filt.shape[0]:
        raise ValueError("len(history)+1[%d] != len(filter)[%d]" %
                         (mc_hist.shape[0] + 1, mc_filt.shape[0]))
    if CYTHON_AVAILABLE is True:
        dtype = mc_data.dtype
        if dtype not in [sp.float32, sp.float64]:
            dtype = sp.float32
        if mc_data.shape[1] != mc_filt.shape[1]:
            raise ValueError("channel count does not match")
        mc_data, mc_filt, mc_hist = (
            sp.ascontiguousarray(mc_data, dtype=dtype),
            sp.ascontiguousarray(mc_filt, dtype=dtype),
            sp.ascontiguousarray(mc_hist, dtype=dtype))
        if dtype == sp.float32:
            return _mcfilter_hist_cy32(mc_data, mc_filt, mc_hist)
        elif dtype == sp.float64:
            return _mcfilter_hist_cy64(mc_data, mc_filt, mc_hist)
        else:
            raise TypeError("dtype is not float32 or float64: %s" % dtype)
    else:
        return _mcfilter_hist_py(mc_data, mc_filt, mc_hist)

## MAIN

if __name__ == "__main__":
    pass

## EOF
