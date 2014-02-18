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


"""general utility functions"""
__docformat__ = "restructuredtext"
__all__ = [
    "sortrows", "vec2ten", "ten2vec", "mcvec_to_conc", "mcvec_from_conc",
    "xcorr", "shifted_matrix_sub", "dict_sort_ndarrays", "get_idx",
    "dict_list_to_ndarray", ]

## IMPORTS

import scipy as sp

## FUNCTIONS

def sortrows(data):
    """sort matrix by rows

    :param ndarray data: array that should be sorted by rows
    :return: ndarray -- data sorted by its rows.
    """

    ## FIX: this method assumes the dta to be continuous! we now make sure of that explicitely
    data = sp.ascontiguousarray(data)
    ## XIF
    return sp.sort(
        data.view([('', data.dtype)] * data.shape[1]), axis=0
    ).view(data.dtype)


def vec2ten(data, nchan=4):
    """converts templates/spikes that are concatenated across the
    channels to tensors that have an extra dim for the channels

    :param ndarray data: input array [templates][vars * channels]
    :param int nchan: channel count
    :return: ndarray -- data converted to tensor [templates][vars][channels]
    :except: ValueError -- if dimensions mismatch
    """

    # init and checks
    if data.ndim == 1:
        data = sp.atleast_2d(data)
    n, dim = data.shape
    if dim % nchan != 0:
        raise ValueError("dim %% nchan != 0 !! dim=%s, nchan=%s" % (dim, nchan))
    tf = dim / nchan
    rval = sp.zeros((n, tf, nchan), data.dtype)

    # convert
    for i in xrange(n):
        for c in xrange(nchan):
            rval[i, :, c] = data[i, c * tf:(c + 1) * tf]
    return rval


def ten2vec(data):
    """converts templates/spikes that are not concatenated across the
    channels to vectors.

    :param ndarray data: input array [templates][vars][channels]
    :return: ndarray -- data converted to concatenated vectors
        [templates][channels * vars]
    """

    # init
    n, tf, nchan = data.shape
    rval = sp.zeros((n, nchan * tf), data.dtype)

    # convert
    for i in xrange(n):
        for c in xrange(nchan):
            rval[i, c * tf:(c + 1) * tf] = data[i, :, c]
    return rval


def mcvec_to_conc(data):
    """convert multi-channeled vector to channel-concatenated vector

    :param ndarray data: multi-channeled vector
    :return: ndarray -- channel-concatenated `data`
    """

    return data.T.flatten()


def mcvec_from_conc(data, nc=4):
    """convert channel-concatenated vector to multi-channeled vector

    :param ndarray data: channel-concatenated vector
    :param int nc: channel count
    :return: ndarray -- multi-channeled `data`
    :except: ValueError -- number of channels mismatch
    """

    ns = data.size / nc
    if ns != round(data.size / nc):
        raise ValueError("nc does not match the vector size!")
    return data.reshape(nc, ns).T


def xcorr(a, b=None, lag=None, normalise=False, unbiased=False):
    """cross-correlation of one-dimensional time series of equal size

    If :b: is not given the auto-correlation of :a: will be computed.

    :param ndarray a: first input time series. ndim=1
    :param ndarray b: second input time series. ndim=1, if None `a` will be taken instead
    :param int lag: lag up to which the cross correlation will be calculated. If
        None all possible lags (2*a.size-1) will be computed.
    :param bool normalise: if True, normalise the result by the size
    :param bool unbiased: if True && normalise is True, use a.size-|tau| to normalize instead of a.size
    :return: ndarray -- cross-correlation of `a` and `b` up to lags `lag`
    """

    # init and check
    a = sp.asarray(a)
    if b is None:
        b = a
    else:
        b = sp.asarray(b)
    if not (a.ndim == b.ndim == 1):
        raise ValueError("a.ndim != b.ndim != 1")
    if a.size != b.size:
        raise ValueError("a.size != b.size")
    if a.size < 2:
        raise ValueError("a.size < 2")
    if lag is None:
        lag = int(a.size - 1)
    if lag > a.size - 1:
        raise ValueError("lag > vector size - 1")
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
    """cross-correlation of one-dimensional time series of equal size (vectorised version)

    :param ndarray a: first input time series. ndim=1
    :param ndarray b: second input time series. ndim=1, if None `a` will be taken instead
    :param int lag: lag up to which the cross correlation will be calculated. If
        None all possible lags (2*a.size-1) will be computed.
    :param dtype dtype: numpy.dtype
    :return: ndarray -- cross-correlation of `a` and `b` up to lags `lag`
    :except: ValueError -- a.ndim != b.ndim or lag > len(vector) - 1
    """

    # init and check
    a = sp.asarray(a)
    b = sp.asarray(b or a)
    if not (a.ndim == b.ndim):
        raise ValueError("a.ndim !== b.ndim")
    if lag is None:
        lag = int(a.shape[0] - 1)
    if lag > a.shape[0] - 1:
        raise ValueError('lag > vector len - 1')
    lag_range = xrange(int(-lag), int(lag + 1))
    rval = sp.empty((a.shape[1], b.shape[1], len(lag_range)), dtype=dtype or a.dtype)

    # calc
    for tau in lag_range:
        prod = a.T[:, None, max(0, +tau):min(len(a), len(a) + tau)] * \
               b.T[None, :, max(0, -tau):min(len(b), len(b) - tau)].conj()
        rval[..., lag + tau] = prod.mean(axis=-1)

    # return
    return rval


def shifted_matrix_sub(data, sub, tau, pad_val=0.0):
    """Subtracts the multi-channeled vector (rows are channels) `sub` from
    the vector `data` with a certain offset. `data` and `sub` may only overlap
    in part, due to the offset.

    REM: from matlab

    :param ndarray data: data array to apply the subtractor to
    :param ndarray sub: sub tractor array
    :param int tau: offset of sub w.r.t. start of data
    :param float pad_val: value to use for the padding
    :return: ndarray -- data minus sub at offset, len(data)
    :except: ValueError -- data minus sub at offset, len(data)
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


def get_idx(idxset, append=False):
    """yields the first available index in a positive integer index set

    :param bool append: if True, returns max(`idxset`)+1, else find the first free index in `idxset`
    :return: int -- first available index
    """

    try:
        idxmax = max(idxset) + 1
        if append is True:
            return idxmax
        idxrange = sp.arange(idxmax)
        return idxrange[sp.nanargmin(sp.in1d(idxrange, idxset))]
    except:
        return 0

## MAIN

if __name__ == "__main__":
    pass

## EOF
