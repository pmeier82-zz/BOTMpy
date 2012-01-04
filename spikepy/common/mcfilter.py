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


"""implementation of mc filter application with time domain FIR filters

There is a general implementation using scipy.correlate suitable for larger
chunks and batch mode filtering. Also there is a filtering strategy that
implements a history item for the part of the signal that could not be
filtered. This implementation is given in Python and in C, the faster variant
(the C implementation) will be used if the libratry providing it can be found
and loaded correctly, with the Python implementation as the fall back.
"""
__docformat__ = 'restructuredtext'
__all__ = ['mcfilter', 'mcfilter_hist', ]

##---IMPORTS

import os
import platform
from ctypes import CDLL, c_uint, c_ushort
import scipy as sp

##---LIBRARY

try:
    _LIB_NAME
    _LIB_PATH
    _LIB_HANDLE
    _RUNTIME_DIR
    _mcfilter_hist_c
except:
    _LIB_NAME = 'SpikePyHelper.dll'
    if platform.system() == 'Linux':
        _LIB_NAME = 'libSpikePyHelper.so'
    _LIB_PATH = None
    _LIB_HANDLE = None
    _RUNTIME_DIR = None
    _mcfilter_hist_c = None
    try:
        _RUNTIME_DIR = os.environ['SPIDAQ_RUNTIME_DIR']
        _LIB_PATH = os.path.join(_RUNTIME_DIR, _LIB_NAME)
        os.chdir(_RUNTIME_DIR)
        _LIB_HANDLE = CDLL(_LIB_PATH)
        _LIB_HANDLE.mcfilter_hist.argtypes = [
            sp.ctypeslib.ndpointer(dtype=sp.float32, ndim=2,
                                   flags="C_CONTIGUOUS"),
            c_uint,
            sp.ctypeslib.ndpointer(dtype=sp.float32, ndim=2,
                                   flags="C_CONTIGUOUS"),
            c_ushort, c_ushort,
            sp.ctypeslib.ndpointer(dtype=sp.float32, ndim=1,
                                   flags="C_CONTIGUOUS,WRITEABLE"),
            sp.ctypeslib.ndpointer(dtype=sp.float32, ndim=2,
                                   flags="C_CONTIGUOUS,WRITEABLE"),
            ]
        _LIB_HANDLE.mcfilter_hist.restype = None
        print 'IMPORT SUCCESS:', _LIB_HANDLE
    except Exception, ex:
        print 'IMPORT ERROR:', repr(ex)


##---FUNCTIONS

def mcfilter(mc_data, mc_fir, correlate_mode='same'):
    """filter a multichanneled signal with a multichanneled filter

    This is the python implementation for batch mode filtering.

    We do not need to account for zero padding, as we are only interested in
     the
    'same' size vector of the xcorr.

    :Parameters:
        mc_data : ndarray
            Data for one channel per columsp.
        mc_fir : ndarray
            A multichanneled finite impulse response filter with either:
            channels concatenate or the filter for each channel on one column.
        correlate_mode : str
            string to pass to scipy.correlate
            Default='same'
    :Returns:
        ndarray
            filtered signal (same shape as data)
    """

    # checks and inits
    if mc_data.ndim != mc_fir.ndim > 2:
        raise ValueError('wrong dimensions: %s, %s' %
                         (mc_data.shape, mc_fir.shape))
    if mc_data.ndim == 1:
        mc_data = sp.atleast_2d(mc_data).T
    nc = mc_data.shape[1]
    if mc_fir.ndim == 1:
        mc_fir = mcvec_from_conc(mc_fir, nc=nc)
    if mc_data.shape[1] != mc_fir.shape[1]:
        raise ValueError('channel count does not match')

    # filter the signal
    return sp.sum(
        [sp.correlate(mc_data[:, c], mc_fir[:, c], mode=correlate_mode) for c
         in xrange(nc)],
                       axis=0
    )


def _mcfilter_hist_py(mc_data, mc_fir, mc_hist=None):
    """filter a multichanneled signal with a multichanneled filter

    We dont need to account for zero padding, as we are only interested in the
    'same' size vector of the xcorr.

    :Parameters:
        mc_data : ndarray
            Data for one channel per columsp.
        mc_fir : ndarray
            A multichanneled finite impulse response filter with either:
            channels concatenate or the filter for each channel on one column.
        hist_item : ndarray
            data history to prepend to the data for filter
    :Returns:
        ndarray
            filtered signal (same shape as data)
        ndarray
            new history item for next filter step
    """

    # checks and inits
    if mc_data.ndim != mc_fir.ndim > 2:
        raise ValueError('wrong dimensions: %s, %s' %
                         (mc_data.shape, mc_fir.shape))
    if mc_data.ndim == 1:
        mc_data = sp.atleast_2d(mc_data).T
    td, nc = mc_data.shape
    if mc_fir.ndim == 1:
        mc_fir = mcvec_from_conc(mc_fir, nc=nc)
    if mc_data.shape[1] != mc_fir.shape[1]:
        raise ValueError('channel count does not match')
    tf = mc_fir.shape[0]
    if mc_hist is None:
        mc_hist = sp.zeros((tf - 1, nc))
    th = mc_hist.shape[0]
    if th + 1 != tf:
        raise ValueError(
            'len(history)+1[%d] != len(filter)[%d]' % (th + 1, tf))
    mc_data = sp.vstack((mc_hist, mc_data))
    rval = sp.zeros(td)

    # filter the signal (by hand)
    for t in xrange(td):
        for c in xrange(nc):
            rval[t] += sp.dot(mc_data[t:t + tf, c], mc_fir[:, c])

    # return
    return rval, mc_data[t + 1:, :].copy()

if _LIB_HANDLE is not None:
    def _mcfilter_hist_c(mc_data, mc_fir, mc_hist):
        """produce the history dependant cross-corelation of the mc_fir with
         the
        mc_data, prepending mc_hist befor the data. a new history item will
        also
        be produced. filter output is written to fout
        """

        # inits
        mc_fout = sp.zeros(mc_data.shape[0], dtype=sp.float32, order='C')
        # XXX: for now we leave these convert here
        mc_data = sp.ascontiguousarray(mc_data, dtype=sp.float32)
        mc_fir = sp.ascontiguousarray(mc_fir, dtype=sp.float32)
        mc_hist = sp.ascontiguousarray(mc_hist, dtype=sp.float32)
        # invoke bindings
        _LIB_HANDLE.mcfilter_hist(mc_data, mc_data.shape[0],
                                  mc_fir, mc_fir.shape[1], mc_fir.shape[0],
                                  mc_fout, mc_hist)

        return mc_fout, mc_hist
mcfilter_hist = _mcfilter_hist_c or _mcfilter_hist_py

## tests

def mcfilter_hist_py_test(inp=None, plot=False):
    if inp is None:
        # test setup
        TF = 10
        NC = 2
        xi = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi, TF))] * NC).T * 5
        LEN = 2000
        noise = sp.randn(LEN, NC)

        # build signal
        signal = sp.zeros_like(noise)
        NPOS = 3
        POS = [int(i * LEN / (NPOS + 1)) for i in xrange(1, NPOS + 1)]
        for i in xrange(NPOS):
            signal[POS[i]:POS[i] + TF] += xi
        x = signal + noise
    else:
        x, xi = inp
        TF, NC = xi.shape
    ns = x.shape[0]

    step = 200
    chunks = [x[i * step:(i + 1) * step] for i in xrange(ns / step)]
    fouts = []
    h = None
    for chunk in chunks:
        r, h = _mcfilter_hist_py(chunk, xi, h)
        fouts.append(r)

    if plot:
        from plot import mcdata

        other = sp.atleast_2d(sp.concatenate(fouts)).T
        other = sp.vstack([other, sp.zeros((int(TF / 2 - 1), 1))])[
                int(TF / 2 - 1):, :]
        mcdata(x, other=other)


def mcfilter_hist_c_test(inp=None, plot=False):
    if _mcfilter_hist_c is None:
        print 'No clib loaded! returning'
        return

    if inp is None:
        # test setup
        TF = 10
        NC = 2
        xi = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi, TF))] * NC).T * 5
        LEN = 2000
        noise = sp.randn(LEN, NC)

        # build signal
        signal = sp.zeros_like(noise)
        NPOS = 3
        POS = [int(i * LEN / (NPOS + 1)) for i in xrange(1, NPOS + 1)]
        for i in xrange(NPOS):
            signal[POS[i]:POS[i] + TF] += xi
        x = signal + noise
    else:
        x, xi = inp
    ns = x.shape[0]

    step = 200
    chunks = [x[i * step:(i + 1) * step] for i in xrange(ns / step)]
    fouts = []
    h = sp.zeros((xi.shape[0], xi.shape[1]), dtype=sp.float32)
    #    r = sp.array([0] * ns, dtype=sp.float32)
    for chunk in chunks:
        r, h = _mcfilter_hist_c(chunk, sp.ascontiguousarray(xi), h)
        fouts.append(r)

    if plot:
        from plot import mcdata

        mcdata(x, other=sp.atleast_2d(sp.concatenate(fouts)).T)


def gen_data(ns=200000, nc=4, tf=65):
    # test setup
    xi = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi, tf))] * nc).T * 7

    signal = sp.randn(ns, nc).astype(sp.float32)
    # build signal
    pos = [50 + i  for i in xrange(1, ns, 4 * tf - 50)]
    if pos[-1] + tf > ns:
        pos.pop(-1)
    for i in xrange(len(pos)):
        signal[pos[i]:pos[i] + tf, :] += xi

    return signal, tf, nc, xi.astype(sp.float32)

if __name__ == '__main__':
    # generate some data
    sig, tf, nc, xi = gen_data(64000)

    # python conventional test
    mcfilter_hist_py_test((sig, xi), plot=True)
    mcfilter_hist_c_test((sig, xi), plot=True)

#    import cProfile
#    cProfile.run('mcfilter_hist_py_test((sig, xi), plot=False)')
#    cProfile.run('mcfilter_hist_c_test((sig, xi), plot=False)')
