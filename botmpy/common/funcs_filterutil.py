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


"""filter related utility functions

!! multi-channeled crosscorrelation filtering implementations are outsourced
to botmpy.mcfilter !!
"""
__docformat__ = "restructuredtext"
__all__ = ["xi_vs_f", "kteo", "mteo"]

## IMPORTS

import scipy as sp
from .funcs_general import mcvec_from_conc
from .funcs_spike import get_cut
from .util import log
from ..mcfilter import mcfilter

## FUNCTIONS

def xi_vs_f(xi, f, nc=4):
    """cross-correlation-tensor for a set of matched patterns and filters

    The xcorr-tensor for a set of patterns (xi) and their matched filters (f)
    with a certain lag is returned as ndarray with dimensions [xi, f, tau].
    All multi-channeled vectors are presented in their concatenated form.

    :type xi: ndarray
    :param xi: The patterns, one concatenated pattern per row.
    :type f: ndarray
    :param f: The filters, one concatenated filter per row.
    :type nc: int
    :param nc: The channel count for the concatenated patterns and filters.

        Default=4
    :returns: ndarray - The tensor of cross-correlation for each pattern
        with each filter. Dimensions as [xi, f, xcorr].
    """
    # init and checks
    xi = sp.asarray(xi)
    f = sp.asarray(f)
    if xi.shape[0] != f.shape[0]:
        raise ValueError("count of xi and f does not match: xi(%s), f(%s)"
                         % (xi.shape[0], f.shape[0]))
    if xi.shape[1] != f.shape[1]:
        raise ValueError("sample count mismatch: xi(%s), f(%s)"
                         % (xi.shape[1], f.shape[1]))
    n = xi.shape[0]
    tf = int(xi.shape[1] / nc)
    if tf != round(float(xi.shape[1]) / float(nc)):
        raise ValueError("sample count does not match to nc: xi(%s), nc(%s)" %
                         (xi.shape[1], nc))
    pad_len = get_cut(tf)[0]
    pad = sp.zeros((pad_len, nc))
    rval = sp.zeros((n, n, 2 * tf - 1))

    # calculation
    for i in xrange(n):
        xi_i = sp.vstack((pad, mcvec_from_conc(xi[i], nc=nc), pad))
        for j in xrange(n):
            f_j = sp.vstack((pad, mcvec_from_conc(f[j], nc=nc), pad))
            rval[i, j] = mcfilter(xi_i, f_j)

    # return
    return rval

## teager energy operator functions

def mteo(data, kvalues=[1, 3, 5], condense=True):
    """multiresolution teager energy operator using given k-values [MTEO]

    The multi-resolution teager energy operator (MTEO) applies TEO operators
    of varying k-values and returns the reduced maximum response TEO for each
    input sample.

    To assure a constant noise power over all kteo channels, we convolve the
    individual kteo responses with a window:
    h_k(i) = hamming(4k+1) / sqrt(3sum(hamming(4k+1)^2) + sum(hamming(4k+1))
    ^2), as suggested in Choi et al., 2006.

    :type data: ndarray
    :param data: The signal to operate on. ndim=1
    :type kvalues: list
    :param kvalues: List of k-values to run the kteo for. If you want to give
        a single k-value, either use the kteo directly or put it in a list
        like [2].
    :type condense: bool
    :param condense: if True, use max operator condensing onto one time series,
        else return a multichannel version with one channel per kvalue.
        Default=True
    :return: ndarray- Array of same shape as the input signal, holding the
        response of the kteo which response was maximum after smoothing for
        each sample in the input signal.
    """
    # init
    rval = sp.zeros((data.size, len(kvalues)))

    # calculation
    for i, k in enumerate(kvalues):
        try:
            rval[:, i] = kteo(data, k)
            win = sp.hamming(4 * k + 1)
            win /= sp.sqrt(3 * (win ** 2).sum() + win.sum() ** 2)
            rval[:, i] = sp.convolve(rval[:, i], win, "same")
        except:
            rval[:, i] = 0.0
            log.warning("MTEO: could not calculate kteo for k=%s, data-length=%s",
                        k, data.size)
    rval[:max(kvalues), i] = rval[-max(kvalues):, i] = 0.0

    # return
    if condense is True:
        rval = rval.max(axis=1)
    return rval


def kteo(data, k=1):
    """teager energy operator of range k [TEO]

    The discrete teager energy operator (TEO) of window size k is defined as:
    M{S{Psi}[x(n)] = x^2(n) - x(n-k) x(n+k)}

    :type data: ndarray
    :param data: The signal to operate on. ndim=1
    :type k: int
    :param k: Parameter defining the window size for the TEO.
    :return: ndarray - Array of same shape as the input signal, holding the
        kteo response.
    :except: If inconsistant dims or shapes.
    """
    # init and checks
    if data.ndim != 1:
        raise ValueError("ndim != 1! ndim=%s with shape=%s" % (data.ndim, data.shape))

    # calculation
    rval = data ** 2 - sp.concatenate((
        [0] * sp.ceil(k / 2.0),
        data[:-k] * data[k:],
        [0] * sp.floor(k / 2.0)))

    # return
    return rval

## MAIN

if __name__ == "__main__":
    pass

## EOF
