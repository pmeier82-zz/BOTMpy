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

"""scaling of multi channeled input data to assert normal background"""
__docformat__ = 'restructuredtext'
__all__ = ['mad_scaling', 'mad_scale_op_mx', 'mad_scale_op_vec']

##  IMPORTS

import scipy as sp
from scipy.stats import norm

## CONSTANTS

NORM_PPF_CONST = 1. / norm.ppf(0.75)

## FUNCTIONS

def _mad(data, center=None, constant=None, axis=0):
    """calculate the median average deviation for multi channel input

    :param ndarray data: multi channeled input data [sample, channel]
    :param float|ndarray center: will be used to calculate the residual in X,
    if None use the median of X

        Default=None
    :param float constant: constant to bias the result,
    if None use the constant corresponding to a normal distribution

        Default=None
    :param int axis: axis to use for the median calculation

        Default=0
    """

    # init and check
    data = sp.asarray(data)
    ns, nc = data.shape
    center = sp.ones(nc) * (center or sp.median(data, axis=axis))
    return (constant or NORM_PPF_CONST) * sp.median(sp.fabs(data - center),
                                                    axis=axis)


def mad_scaling(data, center=None, constant=None, axis=0):
    """scale multi channeled input s.t. the background is standard normal

    :param scipy.ndarray data: multi channeled input data [sample, channel]
    :param scipy.ndarray center: will be used to calculate the residual in X,
    if None use the median of X

        Default=None
    :param float constant: constant to use for the scale value,
    if None use the constant corresponding to a normal distribution

        Default=None
    :param int axis: axis to use for the median calculation

        Default=0
    """

    data = sp.asarray(data)
    scale = _mad(data, center=center, constant=constant, axis=axis)
    return data / scale, scale


def mad_scale_op_mx(mad, tf):
    """build the operator that applies the mad scale to a concatenated spike"""

    return sp.kron(sp.outer(1.0 / mad, 1.0 / mad), sp.ones((tf, tf)))


def mad_scale_op_vec(mad, tf):
    """build the operator that applies the mad scale to a concatenated spike"""

    return sp.kron(1.0 / mad, sp.ones(tf))

## MAIN

if __name__ == '__main__':
    pass

## EOF
