# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012-2013 Berlin Institute of Technology
# Some rights reserved.
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

"""multi-channeled filter application for FIR filters in the time domain

CYTHON IMPLEMENTATIONS
"""
__docformat__ = "restructuredtext"
__all__ = ["mcfilter_f32", "_myfilter_f64", "mcfilter_hist_f32", "_mc_filter_hist_f64"]

## IMPORTS

import cython

cimport cython

import numpy as np

cimport numpy as np

## INIT

cdef init_cy_mcfilter():
    pass

## MCFILTER

@cython.boundscheck(False)
@cython.wraparound(False)
def mcfilter_f32(
        np.ndarray[np.float32_t, ndim=2] mc_data not None,
        np.ndarray[np.float32_t, ndim=2] mc_filt not None):
    cdef:
        unsigned int nc = mc_data.shape[1]
        unsigned int td = mc_data.shape[0]
        unsigned int tf = mc_filt.shape[0]
        np.ndarray[np.float32_t, ndim=1] fout
        np.ndarray[np.float32_t, ndim=2] data
        np.ndarray[np.float32_t, ndim=2] pad
        np.float32_t value
        unsigned int t, tau, c
    pad = np.zeros((np.floor(tf / 2), nc), dtype=np.float32)
    data = np.vstack((pad, mc_data, pad))
    fout = np.empty(td, dtype=np.float32)
    with nogil:
        for t in range(td):
            value = 0.0
            for c in range(nc):
                for tau in range(tf):
                    value += data[t + tau, c] * mc_filt[tau, c]
            fout[t] = value
    return fout

@cython.boundscheck(False)
@cython.wraparound(False)
def mcfilter_f64(
        np.ndarray[np.float64_t, ndim=2] mc_data not None,
        np.ndarray[np.float64_t, ndim=2] mc_filt not None):
    cdef:
        unsigned int nc = mc_data.shape[1]
        unsigned int td = mc_data.shape[0]
        unsigned int tf = mc_filt.shape[0]
        np.ndarray[np.float64_t, ndim=1] fout
        np.ndarray[np.float64_t, ndim=2] data
        np.ndarray[np.float64_t, ndim=2] pad
        np.float32_t value
        unsigned int t, tau, c
    pad = np.zeros((np.floor(tf / 2), nc), dtype=np.float64)
    data = np.vstack((pad, mc_data, pad))
    fout = np.empty(td, dtype=np.float64)
    with nogil:
        for t in range(td):
            value = 0.0
            for c in range(nc):
                for tau in range(tf):
                    value += data[t + tau, c] * mc_filt[tau, c]
            fout[t] = value
    return fout

## MCFILTER_HIST

@cython.boundscheck(False)
@cython.wraparound(False)
def mcfilter_hist_f32(
        np.ndarray[np.float32_t, ndim=2] mc_data not None,
        np.ndarray[np.float32_t, ndim=2] mc_filt not None,
        np.ndarray[np.float32_t, ndim=2] mc_hist not None):
    cdef:
        unsigned int nc = mc_data.shape[1]
        unsigned int td = mc_data.shape[0]
        unsigned int tf = mc_filt.shape[0]
        unsigned int th = mc_hist.shape[0]
        np.ndarray[np.float32_t, ndim=1] fout
        np.ndarray[np.float32_t, ndim=2] data
        np.float32_t value
        unsigned int t, tau, c
    data = np.vstack((mc_hist, mc_data))
    fout = np.empty(td, dtype=np.float32)
    with nogil:
        for t in range(td):
            value = 0.0
            for c in range(nc):
                for tau in range(tf):
                    value += data[t + tau, c] * mc_filt[tau, c]
            fout[t] = value
        for t in range(th):
            for c in range(nc):
                mc_hist[t, c] = data[td + t, c]
    return fout, mc_hist

@cython.boundscheck(False)
@cython.wraparound(False)
def mcfilter_hist_f64(
        np.ndarray[np.float64_t, ndim=2] mc_data not None,
        np.ndarray[np.float64_t, ndim=2] mc_filt not None,
        np.ndarray[np.float64_t, ndim=2] mc_hist not None):
    cdef:
        unsigned int nc = mc_data.shape[1]
        unsigned int td = mc_data.shape[0]
        unsigned int tf = mc_filt.shape[0]
        unsigned int th = mc_hist.shape[0]
        np.ndarray[np.float64_t, ndim=1] fout
        np.ndarray[np.float64_t, ndim=2] data
        np.float64_t value
        unsigned int t, tau, c
    data = np.vstack((mc_hist, mc_data))
    fout = np.empty(td, dtype=np.float64)
    with nogil:
        for t in range(td):
            value = 0.0
            for c in range(nc):
                for tau in range(tf):
                    value += data[t + tau, c] * mc_filt[tau, c]
            fout[t] = value
        for t in range(th):
            for c in range(nc):
                mc_hist[t, c] = data[td + t, c]
    return fout, mc_hist

## MAIN

if __name__ == "__main__":
    pass

## EOF
