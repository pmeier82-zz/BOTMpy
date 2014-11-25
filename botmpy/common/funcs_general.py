# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012 Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Philipp Meier <pmeier82@gmail.com>
#               Neural Information Processing Group (NI)
#               School for Electrical Engineering and Computer Science
#               Berlin Institute of Technology
#               MAR 5-6, Marchstr. 23, 10587 Berlin, Germany
#               http://www.ni.tu-berlin.de/
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


"""general utility functions"""
__docformat__ = 'restructuredtext'
__all__ = [
    'sortrows', 'vec2ten', 'ten2vec', 'mcvec_to_conc', 'mcvec_from_conc',
    'xcorr', 'shifted_matrix_sub', 'dict_sort_ndarrays',
    'dict_list_to_ndarray', 'get_idx']

##--- IMPORTS

import scipy as sp
from scipy import linalg as sp_la

##---FUNCTIONS

## general array operations

def sortrows(data):
    """sort matrix by rows

    :type data: ndarray
    :param data: ndarray that should be sorted by its rows
    :returns: ndarray - data sorted by its rows.
    """

    ## FIX: this method assumes the data to be continuous! we now make sure of that explicitely
    data = sp.ascontiguousarray(data)
    ## XIF
    return sp.sort(
        data.view([('', data.dtype)] * data.shape[1]), axis=0
    ).view(data.dtype)


def vec2ten(data, nchan=4):
    """converts from templates/spikes that are concatenated across the
    channels to tensors that have an extra dim for the channels

    :type data: ndarray
    :param data: input array [templates][vars * channels]
    :type nchan: int
    :param nchan: count of channels
        Default=4
    :returns: ndarray - data converted to tensor [templates][vars][channels]
    """

    if data.ndim == 1:
        data = sp.atleast_2d(data)
    n, dim = data.shape

    if dim % nchan != 0:
        raise ValueError(
            'dim %s nchan != 0 !! dim=%s, nchan=%s' % (dim, nchan))
    tf = dim / nchan

    rval = sp.zeros((n, tf, nchan), data.dtype)

    for i in xrange(n):
        for c in xrange(nchan):
            rval[i, :, c] = data[i, c * tf:(c + 1) * tf]
    return rval


def ten2vec(data):
    """converts from templates/spikes that are not concatenated across the
    channels to vectors.

    :type data: ndarray
    :param data: input array [templates][vars][channels]
    :returns: ndarray- data converted to concatenated vectors
        [templates][channels * vars]
    """

    # init
    n, tf, nchan = data.shape
    rval = sp.zeros((n, nchan * tf), data.dtype)

    # transform
    for i in xrange(n):
        for c in xrange(nchan):
            rval[i, c * tf:(c + 1) * tf] = data[i, :, c]

    # return
    return rval


def mcvec_to_conc(x):
    """returns the concatenated vector for a multichanneled vector

    :type x: ndarray
    :param x: multi-channeled vector in matrix form
    :returns: ndarray - multi-channeled vector in channel concatenated form
    """

    return x.T.flatten()


def mcvec_from_conc(x, nc=4):
    """returns the multichanneled vector from a concatenated representation

    :type x: ndarray
    :param x: multi-channeled vector in channel concatenated form
    :type nc: int
    :param nc: channel count
    :returns: ndarray - multi-channeled vector in matrix form
    """

    nsamples = x.size / nc
    if nsamples != round(x.size / nc):
        raise ValueError('nc does not match the vector size!')
    return x.reshape(nc, nsamples).T


def xcorr(a, b=None, lag=None, normalise=False, unbiased=False):
    """cross-correlation for one-dimensional input signals of equal size

    If :b: is not given the auto-correlation of :a: will be computed.

    :type a: ndarray
    :param a: one-dimensional time series
    :type b: ndarray
    :param b: one-dimensional time series, if None :a: will be taken instead
        Default=None
    :type lag: int
    :param lag: lag up to which the cross correlation will be calculated. If
        None all possible lags (2*a.size-1) will be computed.
        Default=None
    :type normalise: bool
    :param normalise: if True, normalise
        Default=True
    :type unbiased: bool
    :param unbiased: if True and :normalise: is True, use a.size-|tau| to
        normalize instead of a.size
        Default=False
    :returns: ndarray - cross-correlate of :a: and :b: upt to lags :lag:
    """

    # checks
    a = sp.asarray(a)
    if b is None:
        b = a
    else:
        b = sp.asarray(b)
    if not (a.ndim == b.ndim == 1):
        raise ValueError('a.ndim != b.ndim != 1')
    if a.size != b.size:
        raise ValueError('a.size != b.size')
    if a.size < 2:
        raise ValueError('a.size < 2')
    if lag is None:
        lag = int(a.size - 1)
    if lag > a.size - 1:
        raise ValueError('lag > vector size - 1')

    # init
    T = a.size
    lag_range = xrange(int(-lag), int(lag) + 1)
    rval = sp.empty(len(lag_range), dtype=a.dtype)

    # calc
    for tau in lag_range:
        rval[lag + tau] = sp.dot(a[max(0, +tau):min(T, T + tau)], b[max(0, -tau):min(T, T - tau)])

    # normalise
    if normalise is True:
        denom = sp.array([T] * len(lag_range))
        if unbiased is True:
            denom -= sp.absolute(lag_range)
        rval /= denom

    # return
    return rval


def xcorrv(a, b=None, lag=None, dtype=None):
    """vectorial cross correlation by taking the expectation over an outer product"""

    # checks
    a = sp.asarray(a)
    b = sp.asarray(b or a)
    if not (a.ndim == b.ndim):
        raise ValueError('a.ndim !== b.ndim')

    #if a.size != b.size:
    #    raise ValueError('a.size != b.size')
    #if a.size < 2:
    #    raise ValueError('a.size < 2')

    if lag is None:
        lag = int(a.shape[0] - 1)
    if lag > a.shape[0] - 1:
        raise ValueError('lag > vector len - 1')

    # init
    lag_range = xrange(int(-lag), int(lag) + 1)
    rval = sp.empty((a.shape[1], b.shape[1], len(lag_range)), dtype=dtype or a.dtype)

    # calc
    for tau in lag_range:
        prod = a.T[:, None, max(0, +tau):min(len(a), len(a) + tau)] * \
               b.T[None, :, max(0, -tau):min(len(b), len(b) - tau)].conj()
        rval[..., lag + tau] = prod.mean(axis=-1)

    # return
    return rval

## filtering and related processing

def shifted_matrix_sub(data, sub, tau, pad_val=0.0):
    """Subtracts the multi-channeled vector (rows are channels) y from
    the vector x with a certain offset. x and y can due to the offset be only
    partly overlapping.

    REM: from matlab

    :type data: ndarray
    :param data: data array to apply the subtractor to
    :type sub: ndarray
    :param sub: subtractor array
    :type tau: int
    :param tau: offset of :sub: w.r.t. start of :data:
    :type pad_val: float
    :param pad_val: value to use for the padding
        Default=0.0
    :return: ndarray - data minus sub at offset, len(data)
    """

    ns_data, nc_data = data.shape
    ns_sub, nc_sub = sub.shape
    if nc_data != nc_sub:
        raise ValueError('nc_data and nc_sub must agree!')
    tau = int(tau)
    data_sub = sp.empty_like(data)
    data_sub[:] = pad_val
    data_sub[max(0, tau):tau + ns_sub] = sub[max(0, -tau):ns_data - tau]
    return data - data_sub

## data structure utilities

def dict_list_to_ndarray(in_dict):
    """converts all lists in a dictionary to ndarray, works recursively

    :type in_dict: dict
    :param in_dict: dict to workd on
    :returns: dict - all list are converted to ndarray
    """

    for k in in_dict:
        if isinstance(in_dict[k], list):
            in_dict[k] = sp.asarray(in_dict[k])
        elif isinstance(in_dict[k], dict):
            in_dict[k] = dict_list_to_ndarray(in_dict[k])
        else:
            pass
    return in_dict


def dict_sort_ndarrays(in_dict):
    """sort all arrays in a dictionary, works recursively

    :type in_dict: dict
    :param in_dict: dict to work on
    :returns:
    """

    for k in in_dict:
        if isinstance(in_dict[k], sp.ndarray):
            # in_dict[k] = sp.sort(in_dict[k])
            in_dict[k] = sp.sort(in_dict[k])
        elif isinstance(in_dict[k], dict):
            in_dict[k] = dict_sort_ndarrays(in_dict[k])
        else:
            pass
    return in_dict

## index calculations

def get_idx(idxset, append=False):
    """yields the first free index in a positive integer index set

    :type append: bool
    :param append: if True, returns max(:idxset:)+1,
        else find the first free index in :idxset:
    :returns: int - the first free index
    """

    try:
        idxmax = max(idxset) + 1
        if append is True:
            return idxmax
        idxrange = sp.arange(idxmax)
        return idxrange[sp.nanargmin(sp.in1d(idxrange, idxset))]
    except:
        return 0

##---MAIN

if __name__ == '__main__':
    pass
