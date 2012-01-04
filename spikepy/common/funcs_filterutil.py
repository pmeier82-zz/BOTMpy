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


"""filter related utility functions

!! filtering implementations are outsourced to common.mcfilter.py !!
"""
from spikepy.common import mcfilter

__docformat__ = 'restructuredtext'
__all__ = ['xi_vs_f', 'kteo', 'mteo']

##--- IMPORTS

import scipy as sp
from .funcs_general import mcvec_from_conc

##---FUNCTIONS

def xi_vs_f(xi, f, nc=4):
    """cross-correlation-tensor for a set of matched patterns and filters

    The xcorr-tensor for a set of patterns (xi) and their matched filters (f)
    with a certain lag is returned as ndarray with dimensions [xi, f, tau].
    All multichanneled vectors are presented in their concatenated form.

    :Parameters:
        xi : ndarray
            The patterns, one concatenated pattern per row.
        f : ndarray
            The filters, one concatenated filter per row.
        nc : int
            The channel count for the concatenated patterns and filters.
            Default=4
    :Returns:
        ndarray
            The tensor of cross-correlation for each pattern with each filter.
            Dimensions as [xi, f, xcorr].
    """

    # inits and checks
    xi_ = sp.asarray(xi)
    f_ = sp.asarray(f)
    if xi_.shape[0] != f_.shape[0]:
        raise ValueError('count of xi an f does not match: xi(%s), f(%s)'
        % (xi_.shape[0], f_.shape[0]))
    if xi_.shape[1] != f_.shape[1]:
        raise ValueError('sample count mismatch: xi(%s), f(%s)'
        % (xi_.shape[1], f_.shape[1]))
    n = xi_.shape[0]
    tf = int(xi_.shape[1] / nc)
    if tf != round(xi_.shape[1] / nc):
        raise ValueError('sample count does not match to nc: xi(%s), nc(%s)'
        % (xi_.shape[1], nc))
    rval = sp.zeros((n, n, 2 * tf - 1))
    # calc xcorrs
    for i in xrange(n):
        xi_i = mcvec_from_conc(xi_[i], nc=nc)
        for j in xrange(n):
            f_j = mcvec_from_conc(f_[j], nc=nc)
            rval[i, j] = mcfilter(xi_i, f_j, 'full')

    # return
    return rval

## teager energy operator functions

def mteo(X, kvalues=[1, 3, 5], condense=True):
    """multiresolution teager energy operator using given k-values [MTEO]

    The multiresolution teager energy operator (MTEO) applies TEO operators of
    varying k-values and returns the maximum response TEO for each input
    sample.

    To assure a constant noise power over all kteo channels, we convolve the
    individual kteo responses with a window:
    h_k(i) = hamming(4k+1) / sqrt(3sum(hamming(4k+1)^2) + sum(hamming(4k+1))
    ^2),
    as suggested in Choi et al., 2006.

    :Parameters:
        X : ndarray
            The signal to operate on. ndim=1
        kvalues : list
            List of k-values to run the kteo for. If you want to give a single
            k-value, either use the kteo directly or put it in a list like
            [2].
    :Returns:
        ndarray
            Array of same shape as the input signal, holding the response of
             the
            kteo which response was maximum after smoothing for each sample in
            the input signal.
    """

    # inits
    rval = sp.zeros((X.size, len(kvalues)))

    # evaluate the kteos
    for i in xrange(len(kvalues)):
        k = kvalues[i]
        rval[:, i] = kteo(X, k)
        win = sp.hamming(4 * k + 1)
        win /= sp.sqrt(3 * (win ** 2).sum() + win.sum() ** 2)
        rval[:, i] = sp.convolve(rval[:, i], win, 'same')
    rval[:max(kvalues), i] = rval[-max(kvalues):, i] = 0.0

    # return
    if condense is True:
        rval = rval.max(axis=1)
    return rval


def kteo(X, k=1):
    """teager energy operator of range k [TEO]

    The discrete teager energy operator (TEO) of window size k is defined as:
    M{S{Psi}[x(n)] = x^2(n) - x(n-k) x(n+k)}

    :Parameters:
        X : ndarray
            The signal to operate on. ndim=1
        k : int
            Parameter defining the window size for the TEO.
    :Returns:
        ndarray
            Array of same shape as the input signal,
            holding the kteo response.
    :Exception ValueError: If inconsistant dims or shapes.
    """

    # checks and inits
    if X.ndim != 1:
        raise ValueError(
            'ndim != 1! ndim=%s with shape=%s' % (X.ndim, X.shape))

    # apply nonlinear energy operator with range k
    rval = X ** 2 - sp.concatenate(([0] * sp.ceil(k / 2.0),
                                    X[:-k] * X[k:],
                                    [0] * sp.floor(k / 2.0)))

    # return
    return rval

##--- MAIN

def xvf_test():
    from spikepy.common import mcvec_to_conc
    from spikeplot import plt

    xi1 = sp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 6, 4, 2, 0]] * 2,
                   dtype=float).T
    xi2 = sp.array([[0, 2, 4, 6, 8, 7, 6, 5, 4, 3, 2, 1, 0]] * 2,
                   dtype=float).T
    xc = mcfilter(xi2, xi1)
    print xi1.shape, xi2.shape, xc.shape
    plt.plot(xc)
    plt.show()

    xis = sp.asarray([mcvec_to_conc(xi1), mcvec_to_conc(xi2)])
    print xis.shape
    xvf = xi_vs_f(xis, xis, nc=2)
    print xvf

if __name__ == '__main__':
    xvf_test()
