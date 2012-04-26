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

"""spike noise prewhitening algorithm"""

__docformat__ = 'restructuredtext'
__all__ = ['PrewhiteningNode', 'PrewhiteningNode2']

##--- IMPORTS

import scipy as sp
from scipy import linalg as sp_la
from .base_nodes import Node
from ..common import coloured_loading, TimeSeriesCovE

##--- CLASSES

class PrewhiteningNode(Node):
    """prewhitens the data with respect to a noise covariance matrix"""

    ## constructor

    def __init__(self, ncov=None, dtype=sp.float32):
        """
        :Parameters:
            ncov : ndarray
                The noise covariance matrix or None
        """

        # super
        super(PrewhiteningNode, self).__init__(dtype=dtype)

        # members
        self._ncov = None
        self._chol_ncov = None
        self._inv_chol_ncov = None
        self._is_ready = False

        # build
        if ncov is not None:
            self.update(ncov)

    ## privates

    def update(self, ncov):
        """updates the covariance matrix and recalculates internals

        :Parameters:
            ncov : ndarray
                symetric matrix, noise covariance
        """

        # checks
        if ncov.ndim != 2 or ncov.shape[0] != ncov.shape[1]:
            raise ValueError('noise covariance is not a symmetric, '
                             'pos. definite matrix')

        # inits
        self.input_dim = ncov.shape[0]
        self._ncov = ncov
        self._chol_ncov = None
        self._inv_chol_ncov = None

        # compute cholesky decomposition
        try:
            self._chol_ncov = sp_la.cholesky(self._ncov)
        except:
            self._ncov = coloured_loading(self._ncov, 50)
            self._chol_ncov = sp_la.cholesky(self._ncov)
            # invert
        self._inv_chol_ncov = sp_la.inv(self._chol_ncov)

        # set ready flag
        self._is_ready = True

    ## node implementation

    def is_invertable(self):
        return False

    def is_trainable(self):
        return False

    def _execute(self, x, ncov=None):
        # check for update
        if ncov is not None:
            self.update(ncov)

        # ready check
        if self._is_ready is False:
            raise RuntimeError('Node not initialised yet!')

        # return prewhitened data
        return sp.dot(x, self._inv_chol_ncov).astype(self.dtype)


class PrewhiteningNode2(Node):
    """pre-whitens data with respect to a noise covariance matrix"""

    ## constructor

    def __init__(self, covest):
        """
        :type covest: TimeSeriesCovE
        :param covest: noise covariance estimator
        """

        # checks
        if not isinstance(covest, TimeSeriesCovE):
            raise TypeError('expecting instance of TimeSeriesCovE!')

        # super
        super(PrewhiteningNode2, self).__init__(dtype=covest.dtype)

        # members
        self._covest = covest

    ## node implementation

    def is_invertable(self):
        return True

    def is_trainable(self):
        return False

    def _execute(self, x):
        if self._covest.is_initializes is False:
            raise RuntimeError('Node not initialised yet!')

        # return prewhitened data
        rval = sp.dot(x, self._covest.get_whitening_op(tf=self.input_dim))
        return rval.astype(self.dtype)

##--- MAIN

if __name__ == '__main__':
    pass
