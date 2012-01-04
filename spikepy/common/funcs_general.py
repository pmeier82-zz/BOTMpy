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


"""utiliy functions"""
__docformat__ = 'restructuredtext'
__all__ = ['dict_sort_ndarrays', 'dict_list_to_ndarray', 'filter_calculation',
           'find_sorting', 'matrix_argmax', 'matrix_argmin', 'matrixrank',
           'princomp', 'shift_rows', 'shifted_matrix_mul', 'sortrows',
           'shifted_matrix_sub', 'ten2vec', 'vec2ten', 'xcorr',
           'mcvec_from_conc', 'mcvec_to_conc', 'get_idx']

##--- IMPORTS

from scipy import linalg as sp_la
from spikepy.common.constants import *

##---FUNCTIONS

## general array operations

def sortrows(data):
    """sort matrix by rows

    :Parameters:
        data : ndarray
            the ndarray that should be sorted by its rows
    :Returns:
        ndarray
            data sorted by its rows.
    """

    return sp.sort(
        data.view([('', data.dtype)] * data.shape[1]), axis=0
    ).view(data.dtype)


def vec2ten(data, nchan=4):
    """converts from templates/spikes that are concatenated across the
    channels
    to tensors that have an extra dim for the channels

    :Parameters:
        data : ndarray
            input array [templates][vars * channels]
        nchan : int
            count of channels
            Default=4
    :Returns:
        ndarray
            data converted to tensor [templates][vars][channels]
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

    :Parameters:
        data : ndarray
            input array [templates][vars][channels]
    :Returns:
        ndarray
            data converted to concatenated vectors [templates][channels *
            vars]
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
    """returns the concatenated vector for a multichanneled vector"""

    return x.T.flatten()


def mcvec_from_conc(x, nc=4):
    """returns the multichanneled vector from a concatenated representation"""

    nsamples = x.size / nc
    if nsamples != round(x.size / nc):
        raise ValueError('nc does not match the vector size!')
    return x.reshape(nc, nsamples).T


def xcorr(a, b=None, lag=None, unbiased=False):
    """xcorr, with specific lag and correct normalisation w.r.t.
    zero-padding"""

    # checks
    if b is None:
        b = a
    if not (a.ndim == b.ndim == 1):
        raise ValueError('not a.ndim == b.ndim == 1')
    if a.size < 2:
        raise ValueError('a.size < 2')
    if a.size != b.size:
        raise ValueError('a.size != b.size')
    if lag is None:
        lag = a.size - 1
    if lag > a.size - 1:
        raise ValueError('lag > vector size - 1')

    # inits
    T = a.size
    lagrange = xrange(int(-lag), int(lag) + 1)
    rval = sp.zeros(len(lagrange), dtype=a.dtype)

    # calc
    for tau in lagrange:
        rval[lag + tau] = sp.dot(a[max(0, +tau):min(T, T + tau)],
                                 b[max(0, -tau):min(T, T - tau)])
        norm_fac = T
        if unbiased is True:
            norm_fac -= abs(tau)
        rval[lag + tau] /= norm_fac

    # return
    return rval


def princomp(data, explain=0, percentage=False):
    """calculate the principal component projections for data

    :Parameters:
        data : ndarray
            the signal data with observations in the rows and variables in the
            columns.
        explain : number
            if percentage is False, the data will be projected into the
            explain
            first principal components. If percentage is True, explain} has to
            be a value between 0 and 1 and the C{data} will be projected
            into as
            many principal components as needed to explain explain
            percentage of
            the total variance. If explain is 0, the data will not be
            projected
            at all.
        percentage : float
            value between 0 and 1 as a percentage of the total variance in the
            data that should be explained by the projection.
    :Returns:
        ndarray
            eigenvectors of the covariance matrix (in the columns)
        ndarray
            eigenvalues of the covariance matrix
        ndarray
            data projected into a set of the eigenvectors
        float
            percentage of the total variance explained by the projected data
    """

    raise DeprecationWarning('USE MDP-TOOLKIT NODE INSTEAD!!')

    # checks
    if type(data) is not dict:
        data = {0:data}

    # inits
    dim = data[data.keys()[0]].shape[1]
    data_cov = sp.zeros([dim] * 2)
    total_samples = 0

    # compute data mean
    data_mean = sp.zeros(dim)
    for k in data.keys():
        my_samples = data[k].shape[0]
        my_mean = data[k].mean(axis=0)
        total_samples += my_samples
        data_mean += my_mean * my_samples
    data_mean /= total_samples

    # compute covariance
    for k in data.keys():
        my_samples = data[k].shape[0]
        data_mean_cor = data[k] - data_mean
        data_cov += sp.cov(data_mean_cor.T) * my_samples
    data_cov /= total_samples

    # get eigenvalues/-vectors and sort
    v, pc = sp_la.eig(data_cov)
    v, pc = sp.real(v), sp.real(pc)
    sort_idx = (-v).argsort()
    v, pc = v[sort_idx], pc[:, sort_idx]

    # project data
    pca_data = {}
    explained = None
    if explain > 0:
        max_pc = explain
        if percentage is True:
            if 0 < explain < 1:
                pass
            elif 1 <= explain <= 100:
                explain /= 100.0
            else:
                raise ValueError('misleading explain value %s' % explain)
            explain_var = explain * v.sum()
            max_pc = (v.cumsum() < explain_var).argmin()
        if max_pc < 2:
            max_pc = 2
        for k in data.keys():
            pca_data[k] = sp.dot(data[k], pc[:, :max_pc])
        explained = v[:max_pc].sum() / v.sum()

    return pc, v, pca_data, explained

## filtering and related processing

def filter_calculation(template, invR=None):
    """calculates an optimal linear filter [citation]

    The filter is calculated from the inverse of a the data covariance matrix
    and a (potentially multichanneled) template.

    :Parameters:
        template : ndarray
            The (potentially multichanneled) template, channels on the rows,
            vars on the columns.
        invR : ndarray
            Inverse of the data covariance matrix. If None, identiy matrix is
            substituted.
    """

    # check for invR
    if invR is None:
        invR = sp.eye(template.shape[0], dtype=template.dtype)

    # compute
    return sp.dot(invR, template.T) / sp.dot(sp.dot(template, invR),
                                             template.T)


def shifted_matrix_sub(data, sub, tau, pad_val=0.0):
    """Subtracts the multi channel vector (rows are channels) y from
    the vector x with a certain offset. x and y can due to the offset be only
    partly overlapping

    :type data: ndarray
    :param data: the data array to apply the subtractor to
    :type sub: ndarray
    :param sub: the subtractor array
    :type tau: int
    :param tau: offset to the subtractor w.r.t. start of :data:
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


def shifted_matrix_mul(mix, data, shift, trunc=True):
    """Solves the equation rval = mix * shift(data) where shift() is a
    shift-operator for each row (i) in mix, the rows (j) in data are
    shifted by shift[i] [citation]

    :Parameters:
        mix : ndarray
            Symetric mixture matrix.
        data : ndarray
            Data for one unit vstacked. This is the confused filter output for
            each unit.
        shift : ndarray
            Symetric shifting matrix.
    :Returns:
        ndarray
            the result of mix * shift(data), the deconfused filter output,
            where
            each filter output was replaced by a component-wise shifted
            mixture
            of all given filter outputs, respecting the mixture and shifting
            matrix.
    """

    # checks and inits
    if not data.any():
        raise ValueError('data is a null matrix')
    if data.shape[0] != shift.shape[0] != mix.shape[0]:
        raise ValueError('shape mismatch!')

    min_shift = shift.min()
    max_shift = shift.max()

    rval = sp.zeros(
        (data.shape[0], data.shape[1] - min_shift + max_shift),
                                                              dtype=data.dtype
    )

    # loop over templates to mix outputs of all filters to that template
    for i in xrange(data.shape[0]):
        min_shift_i = shift[i, :].min()
        max_shift_i = shift[i, :].max()

        rval[i, -min_shift + min_shift_i:
        rval[i, :].size - (max_shift - max_shift_i)] =\
        sp.dot(mix[i, :], shift_rows(data, shift[i, :]))

    # truncate rval back to dims of data
    if trunc is True:
        rval = rval[:, 0 - min_shift:rval.shape[1] - max_shift]

    # return
    return rval


def shift_rows(data, tau, nchan=1):
    """shifts the rows of data by tau. data can contain multichannel data,
    with all rows concatinated. If so, the number of channels has to be given
    by nchan

    :Parameters:
        data : ndarray
            The rows to be shifted
        tau : ndarray
            The shiftin operator, 1d array with on integer for each row,
            indicating the samples of shift. Negative means shift left,
            positive
            means shift right.
        nchan : int
            channel count of the data
    :Returns:
        ndarray
            the shifted row data
    """

    # inits
    n, dim_tot = data.shape
    if tau.size < n:
        raise ValueError('size of tau is too small')
    dim = dim_tot / nchan
    if not tau.any():
        return data
    tau = tau.astype(sp.integer)
    min_tau = tau.min()
    tau = tau - min_tau
    max_tau = tau.max()
    dim_tot_new = dim_tot + nchan * max_tau
    dim_neu = dim_tot_new / nchan

    # calc
    rval = sp.zeros((n, dim_tot_new), dtype=data.dtype)
    for i in xrange(n):
        for c in xrange(nchan):
            idx_old = c * dim + sp.arange(dim)
            idx_new = c * dim_neu + sp.arange(dim) + tau[i]
            rval[i, idx_new] = data[i, idx_old]

    # return
    return rval

## matrix utilities

def matrixrank(M, tol=1e-8):
    """computes the rank of a matrix (sloppy)"""

    return sp.sum(sp.where(sp_la.svd(M, compute_uv=0) > tol, 1, 0))


def matrix_argmax(M):
    """returns the indices (row,col) of the maxmum in M

    :Parameters:
        M : ndarray
            ndarray where to find the maximum
    :Returns:
        tuple
            tuple of indices for each dimension of M indicating the max of M.
    """
    idx = sp.nanargmax(M)
    j = int(idx % M.shape[1])
    i = int(sp.floor(idx / M.shape[1]))
    return i, j

    # DO NOT USE THIS VERSION; SINCE IT DOES NOT WORK IF THE EXTREMUM IS NOT
    # UNIQUE!
    # rval = []
    # for i in reversed(xrange(M.ndim)):
    #     rval.append(M.max(axis=i).argmax())
    # return tuple(rval)


def matrix_argmin(M):
    """returns the indices (row,col) of the minimum in M

    :Parameters:
        M : ndarray
            ndarray where to find the minimum
    :Returns:
        tuple
            tuple of indices for each dimension of M indicating the min of M.
    """
    idx = sp.nanargmin(M)
    j = int(idx % M.shape[1])
    i = int(sp.floor(idx / M.shape[1]))
    return i, j

    # DO NOT USE THIS VERSION; SINCE IT DOES NOT WORK IF THE EXTREMUM IS NOT
    # UNIQUE!
    # rval = []
    # for i in reversed(xrange(M.ndim)):
    #     rval.append(M.min(axis=i).argmin())
    # return tuple(rval)


def find_sorting(data):
    """find the sorting of an ndarray

    :Parameters:
        data : ndarray
            ndarray to sort
    :Returns:
        ndarray
            data sorted
    """

    data = data.copy().tolist()
    rval = []
    for item in sorted(data):
        rval.append(data.index(item))
    return rval


def dict_list_to_ndarray(in_dict):
    """converts all lists in a dictionary to ndarray

    If there are dicts found into the in_dict, this will work rekursively.
    """

    for k, v in in_dict.items():
        if isinstance(in_dict[k], list):
            in_dict[k] = sp.asarray(v)
        elif isinstance(in_dict[k], dict):
            dict_list_to_ndarray(in_dict[k])
        else:
            pass
    return in_dict


def dict_sort_ndarrays(in_dict):
    """sort all arrays in a dictionary"""

    for k in in_dict.keys():
        if isinstance(in_dict[k], sp.ndarray):
            in_dict[k] = sp.sort(in_dict[k])

    return in_dict

## index calculations

def get_idx(idxset):
    """yields the first free index in a positive integer index set"""

    #    # checks
    #    if not isinstance(idxset, list):
    #        raise ValueError('only lists are valid index sets')
    #    if len(idxset) == 0:
    #        return 0
    #    rval = 0
    #    for i in xrange(max(idxset) + 2):
    #        if i not in idxset:
    #            rval = i
    #            break
    #    return rval
    try:
        r = arange(max(idxset) + 1)
        return r[sp.nanargmin(sp.in1d(r, idxset))]
    except:
        return 0

##---MAIN

if __name__ == '__main__':
    pass
