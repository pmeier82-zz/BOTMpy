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


"""loading algorithms for matrices"""
__docformat__ = 'restructuredtext'

##---ALL

__all__ = ['matrix_cond', 'diagonal_loading', 'coloured_loading']

##---IMPORTS

import scipy as sp
from scipy import linalg as sp_la

##---CONSTANTS

SUFFICIENT_CONDITION = 50

##---FUNCTIONS

def matrix_cond(mat):
    """yield the matrix condition number w.r.t. l2-norm (using svd)"""

    mat_ = sp.atleast_2d(mat)
    if mat_.ndim != 2:
        raise ValueError('expected matrix')
    if mat_.size == 0:
        raise ValueError('undefined for empty matrix')
    try:
        sv = sp_la.svd(mat_, compute_uv=False)
        return compute_matrix_cond(sv)
    except:
        return sp.inf


def compute_matrix_cond(sv):
    """yield matix condition number w.r.t. l2-norm given a set of singular
    values"""

    sv_ = sp.atleast_1d(sv)
    if sv_.ndim != 1:
        raise ValueError(
            'please provide singular values as list or 1dim array!')
    if sv_.size == 0:
        raise ValueError('undefined for empty list')
    try:
        return sp.absolute(sv_[0] / sv_[-1])
    except:
        return sp.inf


def diagonal_loading(mat, target_cond=SUFFICIENT_CONDITION,
                     overwrite_mat=False):
    """tries to condition the matrix by adding the scaled identity matrix

    solves: cond(mat + alpha*I) = target_cond for alpha

    Note that nothing will be done if the condition is already >= target_cond!

    imposes a spherical constraint on the covariance ellipsoid
    """

    mat_ = sp.atleast_2d(mat)
    if mat_.ndim != 2:
        raise ValueError('expected matrix')
    if mat_.size == 0:
        raise ValueError('undefined for empty matrix')
    svd = sp_la.svd(mat_)
    return compute_diagonal_loading(mat_, svd, target_cond, overwrite_mat)


def compute_diagonal_loading(mat, svd, target_cond=SUFFICIENT_CONDITION,
                             overwrite_mat=False):
    """tries to condition the matrix by adding the scaled identity matrix

    solves: cond(mat + alpha*I) = target_cond for alpha

    Note that nothing will be done if the condition is already >= target_cond!

    imposes a spherical constraint on the matrix
    """

    sv = svd[1]
    if target_cond == 1:
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
    """tries to condition the matrix by adjusting the lowest eigenvalues

    imposes an spherical constraint on a subspace of the covariance ellipsoid
    """

    mat_ = sp.atleast_2d(mat)
    if mat_.ndim != 2:
        raise ValueError('expected matrix')
    if mat_.size == 0:
        raise ValueError('undefined for empty matrix')
    svd = sp_la.svd(mat_)
    return compute_coloured_loading(mat_, svd, target_cond, overwrite_mat)


def compute_coloured_loading(mat, svd, target_cond=SUFFICIENT_CONDITION,
                             overwrite_mat=False):
    """tries to condition the matrix by adjusting the lowest eigenvalues

    imposes an spherical constraint on the subspace of mat contributing most
     to
    the bad conditioning of mat.
    """

    U, sv = svd[0], svd[1]
    if target_cond == 1:
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

##---MAIN

if __name__ == '__main__':
    r = sp.array([1.0, 0.9, 0.8])
    C = sp_la.toeplitz(r)
    cnos = [10, 15.3, 50]

    print 'initial matrix:'
    print C
    print

    for cno in cnos:
        print 'initial condition:', matrix_cond(C)
        print 'target condition:', cno
        print
        Ddiag = diagonal_loading(C, cno)
        Dcol = coloured_loading(C, cno)
        print 'diagonally loaded:', matrix_cond(Ddiag)
        print Ddiag
        print 'coloured loaded:', matrix_cond(Dcol)
        print Dcol
        print

    print 'C matrix:', matrix_cond(C)
    print C
    Cnew = coloured_loading(C, 10, overwrite_mat=True)
    print 'Cnew loaded at condition:', matrix_cond(Cnew)
    print Cnew
    print 'same matrices: C is Cnew', C is Cnew
    print
