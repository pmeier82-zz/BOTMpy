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
warnings.simplefilter("once")

from .mcfilter import \
    mcfilter as mcfilter_py, \
    mcfilter_hist as mcfilter_hist_py

CYTHON_AVAILABLE = False

try:
    import mcfilter_cy

    CYTHON_AVAILABLE = True
except ImportError, ex:
    mcfilter_cy = None
    warnings.warn(str(ex), ImportWarning)
    warnings.warn("Cython implementation not found! Falling back to Python!", ImportWarning)

## FUNCTIONS

def mcfilter(mc_data, mc_filt, force_py=False):
    """filter a multi-channeled signal with a multi-channeled filter

    The signal will be padded with zeros on both ends to overcome filter
    artifacts. Input and output will have same dimensions.

    .. note:
        If you compiled the cython extension, this will use a faster c code
        implementation.

    :param ndarray mc_data: signal data [data_samples, channels]
    :param ndarray mc_filt: FIR filter [filter_samples, channels]
    :param bool force_py: if True, force use of python implementation
    :return: ndarray -- filtered signal [data_samples]
    """

    if mc_data.ndim != 2 or mc_filt.ndim != 2:
        raise ValueError("wrong dimensions: %s, %s" % (mc_data.shape, mc_filt.shape))
    if mc_data.shape[1] != mc_filt.shape[1]:
        raise ValueError("channel count does not match")

    if CYTHON_AVAILABLE is True and force_py is False:
        dtype = mc_data.dtype
        if dtype not in [sp.float32, sp.float64]:
            dtype = sp.float32
        mc_data, mc_filt = (
            sp.ascontiguousarray(mc_data, dtype=dtype),
            sp.ascontiguousarray(mc_filt, dtype=dtype))
        if dtype == sp.float32:
            return mcfilter_cy.mcfilter_f32(mc_data, mc_filt)
        elif dtype == sp.float64:
            return mcfilter_cy.mcfilter_f64(mc_data, mc_filt)
    else:
        return mcfilter_py(mc_data, mc_filt)


def mcfilter_hist(mc_data, mc_filt, mc_hist=None, force_py=False):
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
    :param bool force_py: if True, force use of python implementation
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
            "history has wrong size (expected: %d, actual: %d" %
            (mc_hist.shape[0] + 1, mc_filt.shape[0]))
    dtype = mc_data.dtype
    if CYTHON_AVAILABLE is True and force_py is False:
        mc_data, mc_filt, mc_hist = (
            sp.ascontiguousarray(mc_data, dtype=dtype),
            sp.ascontiguousarray(mc_filt, dtype=dtype),
            sp.ascontiguousarray(mc_hist, dtype=dtype))
        if dtype == sp.float32:
            return mcfilter_cy.mcfilter_hist_f32(mc_data, mc_filt, mc_hist)
        elif dtype == sp.float64:
            return mcfilter_cy.mcfilter_hist_f64(mc_data, mc_filt, mc_hist)
    else:
        return mcfilter_hist_py(mc_data, mc_filt, mc_hist)

## MAIN

if __name__ == "__main__":
    pass

## EOF
