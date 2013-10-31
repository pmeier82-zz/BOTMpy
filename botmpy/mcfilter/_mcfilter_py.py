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

"""multi-channeled filter application for FIR filters in the time domain

PYTHON IMPLEMENTATIONS
"""
__docformat__ = "restructuredtext"
__all__ = ["_mcfilter_py", "_mcfilter_hist_py", ]

## IMPORTS

import scipy as sp

## FUNCTIONS

def _mcfilter_py(mc_data, mc_filt):
    if mc_data.ndim != mc_filt.ndim > 2:
        raise ValueError("wrong dimensions: %s, %s" %
                         (mc_data.shape, mc_filt.shape))
    if mc_data.shape[1] != mc_filt.shape[1]:
        raise ValueError("channel count does not match")
    return sp.sum(
        [sp.correlate(mc_data[:, c], mc_filt[:, c], mode="same")
         for c in xrange(mc_data.shape[1])],
        axis=0)


def _mcfilter_hist_py(mc_data, mc_filt, mc_hist):
    if mc_data.ndim != mc_filt.ndim > 2:
        raise ValueError("wrong dimensions: %s, %s" %
                         (mc_data.shape, mc_filt.shape))
    if mc_data.shape[1] != mc_filt.shape[1]:
        raise ValueError("channel count does not match")
    mc_hist_and_data = sp.vstack((mc_hist, mc_data))
    rval = sp.zeros(mc_data.shape[0], dtype=mc_data.dtype)
    for t in xrange(mc_data.shape[0]):
        for c in xrange(mc_hist_and_data.shape[1]):
            rval[t] += sp.dot(mc_hist_and_data[t:t + mc_filt.shape[0], c], mc_filt[:, c])
    return rval, mc_data[-(mc_hist.shape[0]):].copy()

## MAIN

if __name__ == "__main__":
    pass

## EOF
