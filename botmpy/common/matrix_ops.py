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


"""matrix operations"""
__docformat__ = 'restructuredtext'
__all__ = ['matrix_cond', 'diagonal_loading', 'coloured_loading',
           'matrix_argmax', 'matrix_argmin']

# TODO: should we enforce square matrices for all ops?

##---IMPORTS

import scipy as sp
from scipy import linalg as sp_la

##---CONSTANTS

SUFFICIENT_CONDITION = 50

##---FUNCTIONS

def matrix_pos_def(mat):
    """checks if the matrix is positive definite

    :type mat: ndarray
    :param mat: input matrix
    :returns: True if p.d., False else
    """

    mat = sp.atleast_2d(mat)
    if mat.ndim != 2:
        raise ValueError('expected matrix')
    if mat.size == 0:
        raise ValueError('undefined for empty matrix')
    try:
        sv = sp_la.svd(mat, compute_uv=False)
        return sp.all(sv > 0.0)
    except:
        return False


def matrix_cond(mat):
    """yield the matrix condition number w.r.t. l2-norm (using svd)

    :type mat: ndarray
    :param mat: input matrix
    :returns: float - condition number of :mat: or :inf: on error
    """

    mat = sp.atleast_2d(mat)
    if mat.ndim != 2:
        raise ValueError('expected matrix')
    if mat.size == 0:
        raise ValueError('undefined for empty matrix')
    try:
        sv = sp_la.svd(mat, compute_uv=False)
        return compute_matrix_cond(sv)
    except:
        return sp.inf


def compute_matrix_cond(sv):
    """yield matrix condition number w.r.t. l2-norm given singular values

    :type sv: ndarray
    :param sv: vector of singular values sorted s.t. :sv[i]: >= :sv[i+1]:
    :returns: float - condition number of :mat: or :inf: on error
    """

    sv = sp.atleast_1d(sv)
    if sv.size == 0:
        raise ValueError('undefined for empty list')
    try:
        return sp.absolute(sv[0] / sv[-1])
    except:
        return sp.inf


def diagonal_loading(mat, target_cond=SUFFICIENT_CONDITION,
                     overwrite_mat=False):
    """tries to condition the :mat: by imposing a spherical constraint on the
    covariance ellipsoid (adding alpha*eye)

    solves: cond(mat + alpha*I) = target_cond for alpha

    Note: this is a noop if the condition is already >= target_cond!

    :type mat: ndarray
    :param mat: input matrix
    :type target_cond: float
    :param target_cond: condition number to archive after loading
    :type overwrite_mat: bool
    :param overwrite_mat: if True, operate inplace and overwrite :mat:
    :returns: ndarray - matrix like :mat: conditioned s.t. cond = target_cond
    """

    mat = sp.atleast_2d(mat)
    if mat.size == 0:
        raise ValueError('undefined for empty matrix')
    svd = sp_la.svd(mat)
    return compute_diagonal_loading(mat, svd, target_cond, overwrite_mat)


def compute_diagonal_loading(mat, svd, target_cond=SUFFICIENT_CONDITION,
                             overwrite_mat=False):
    """tries to condition :mat: by imposing a spherical constraint on the
    covariance ellipsoid (adding alpha*eye)

    solves: cond(mat + alpha*I) = target_cond for alpha

    Note: this is a noop if the condition is already >= target_cond!

    :type mat: ndarray
    :param mat: input matrix
    :type svd: tuple
    :param svd: return tuple of svd(:mat:) - consistency will not be checked!
    :type target_cond: float
    :param target_cond: condition number to archive after loading
    :type overwrite_mat: bool
    :param overwrite_mat: if True, operate inplace and overwrite :mat:
    :returns: ndarray - matrix like :mat: conditioned s.t. cond = target_cond
    """

    sv = svd[1]
    if target_cond == 1.0:
        return sp.eye(mat.shape[0], mat.shape[1])
    if target_cond > compute_matrix_cond(sv):
        return mat
    if overwrite_mat is True:
        rval = mat
    else:
        rval = mat.copy()
    alpha = (sv[0] - target_cond * sv[-1]) / (target_cond - 1)
    return rval + alpha * sp.eye(rval.shape[0], rval.shape[1])


def coloured_loading(mat, target_cond=SUFFICIENT_CONDITION,
                     overwrite_mat=False):
    """tries to condition :mat: by inflating the badly conditioned subspace
    of :mat: using a spherical constraint.

    :type mat: ndarray
    :param mat: input matrix
    :type target_cond: float
    :param target_cond: condition number to archive after loading
    :type overwrite_mat: bool
    :param overwrite_mat: if True, operate inplace and overwrite :mat:
    :returns: ndarray - matrix like :mat: conditioned s.t. cond = target_cond
    """

    mat = sp.atleast_2d(mat)
    if mat.size == 0:
        raise ValueError('undefined for empty matrix')
    svd = sp_la.svd(mat)
    return compute_coloured_loading(mat, svd, target_cond, overwrite_mat)


def compute_coloured_loading(mat, svd, target_cond=SUFFICIENT_CONDITION,
                             overwrite_mat=False):
    """tries to condition :mat: by inflating the badly conditioned subspace
    of :mat: using a spherical constraint.

    :type mat: ndarray
    :param mat: input matrix
    :type svd: tuple
    :param svd: return tuple of svd(:mat:) - consistency will not be checked!
    :type target_cond: float
    :param target_cond: condition number to archive after loading
    :type overwrite_mat: bool
    :param overwrite_mat: if True, operate inplace and overwrite :mat:
    :returns: ndarray - matrix like :mat: conditioned s.t. cond = target_cond
    """

    U, sv = svd[0], svd[1]
    if target_cond == 1.0:
        return sp.eye(mat.shape[0])
    if target_cond > compute_matrix_cond(sv):
        return mat
    if overwrite_mat is True:
        rval = mat
    else:
        rval = mat.copy()
    min_s = sv[0] / target_cond
    for i in xrange(sv.size):
        col_idx = -1 - i
        if sv[col_idx] < min_s:
            alpha = min_s - sv[col_idx]
            rval += alpha * sp.outer(U[:, col_idx], U[:, col_idx])
    return rval


def matrix_argmax(mat):
    """returns the indices (row,col) of the maximum value in :mat:

    :type mat: ndarray
    :param mat: input matrix
    :returns: tuple - (row,col) of the maximum value in :mat:
    """

    idx = sp.nanargmax(mat)
    j = int(idx % mat.shape[1])
    i = int(sp.floor(idx / mat.shape[1]))
    return i, j

    # XXX
    # DO NOT USE THIS VERSION; SINCE IT DOES NOT WORK IF THE EXTREMUM IS NOT
    # UNIQUE!
    # fout = []
    # for i in reversed(xrange(mat.ndim)):
    #     fout.append(mat.max(axis=i).argmax())
    # return tuple(fout)


def matrix_argmin(mat):
    """returns the indices (row,col) of the minimum value in :mat:

    :type mat: ndarray
    :param mat: input matrix
    :returns: tuple - (row,col) of the minimum value in :mat:
    """

    idx = sp.nanargmin(mat)
    j = int(idx % mat.shape[1])
    i = int(sp.floor(idx / mat.shape[1]))
    return i, j

    # XXX
    # DO NOT USE THIS VERSION; SINCE IT DOES NOT WORK IF THE EXTREMUM IS NOT
    # UNIQUE!
    # fout = []
    # for i in reversed(xrange(mat.ndim)):
    #     fout.append(mat.min(axis=i).argmin())
    # return tuple(fout)

##---MAIN

if __name__ == '__main__':
    pass
