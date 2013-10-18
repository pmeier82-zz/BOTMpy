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


"""spikes alignment functions"""
__docformat__ = 'restructuredtext'
__all__ = ['sinc_interp1d', 'get_tau_for_alignment', 'get_tau_align_min',
           'get_tau_align_max', 'get_tau_align_energy', 'get_aligned_spikes']

##  IMPORTS

import scipy as sp
from scipy.signal import resample

from .util import INDEX_DTYPE
from .funcs_spike import epochs_from_spiketrain, get_cut, extract_spikes

## FUNCTIONS

def sinc_interp1d(x, s, r):
    """Interpolates `x`, sampled at times `s`
    Output `y` is sampled at times `r`

    inspired from from Matlab:
    http://phaseportrait.blogspot.com/2008/06/sinc-interpolation-in-matlab.html

    :param ndarray x: input data time series
    :param ndarray s: input sampling time series (regular sample interval)
    :param ndarray r: output sampling time series
    :return ndarray: output data time series (regular sample interval)
    """

    # init
    s = sp.asarray(s)
    r = sp.asarray(r)
    x = sp.asarray(x)
    if x.ndim == 1:
        x = sp.atleast_2d(x)
    else:
        if x.shape[0] == len(s):
            x = x.T
        else:
            if x.shape[1] != s.shape[0]:
                raise ValueError('x and s must be same temporal extend')
    if sp.allclose(s, r):
        return x.T
    T = s[1] - s[0]

    # resample
    sincM = sp.tile(r, (len(s), 1)) - sp.tile(s[:, sp.newaxis], (1, len(r)))
    return sp.vstack([sp.dot(xx, sp.sinc(sincM / T)) for xx in x]).T


def get_tau_for_alignment(spikes, align_at):
    """return the per spike offset in samples (taus) of the maximum values to
    the desired alignment sample within the spike waveform.

    :type spikes: ndarray
    :param spikes: stacked mc spike waveforms [ns, tf, nc]
    :type align_at: int
    :param align_at: sample to align the maximum at
    :returns: ndarray - offset per spike
    """

    # checks
    ns, tf, nc = spikes.shape
    if 0 < align_at >= tf:
        return sp.zeros(ns)

    # offsets
    dchan = [spike.max(0).argmax() for spike in spikes]
    tau = [spikes[i, :, dchan[i]].argmax() - align_at for i in xrange(ns)]
    return sp.asarray(tau, dtype=INDEX_DTYPE)

get_tau_align_min = lambda spks, ali: get_tau_for_alignment(-spks, ali)
get_tau_align_max = lambda spks, ali: get_tau_for_alignment(spks, ali)
get_tau_align_energy = lambda spks, ali: get_tau_for_alignment(spks * spks, ali)

def get_aligned_spikes(data, spike_train, align_at=-1, tf=47, mc=True,
                       kind='none', rsf=1., sample_back=True):
    """return the set of aligned spikes waveforms and the aligned spike train

    :type data: ndarray
    :param data: data with channels in the columns
    :type spike_train: ndarray or list
    :param spike_train: spike train of events in data
    :type align_at: int
    :param align_at: align feature at this sample in the waveform
    :type tf: int
    :param tf: temporal extend of the waveform in samples
    :type mc: bool
    :param mc: if True, return mc waveforms, else return concatenated waveforms.
        Default=True
    :type kind: str
    :param kind: String giving the type of alignment to conduct. One of:

        - "max"    - align on maximum of the waveform
        - "min"    - align on minimum of the waveform
        - "energy" - align on peak of energy
        - "none"   - no alignment

        Default='none'
    :type rsf: float
    :param rsf: resampling factor (use integer values of powers of 2)
    :param bool sample_back: if True, resample spikes to original length after resampling
    :rtype: ndarray, ndarray
    :returns: stacked spike events, spike train with events corrected for
        alignment
    """

    # resample?
    if rsf != 1.0:
        data = resample(data, rsf * data.shape[0])
        tf *= rsf
        align_at *= rsf
        spike_train *= rsf

    # init
    cut = align_at, tf - align_at
    ep, st = epochs_from_spiketrain(
        spike_train,
        cut,
        end=data.shape[0],
        with_corrected_st=True)

    # align spikes
    if ep.shape[0] > 0:
        if kind in ['min', 'max', 'energy']:
            spikes = extract_spikes(data, ep, mc=True)
            if rsf != 1.0:
                print spikes.shape
            tau = {'min': get_tau_align_min,
                   'max': get_tau_align_max,
                   'energy': get_tau_align_energy}[kind](spikes, align_at)
            st += tau

            ep, st = epochs_from_spiketrain(
                st,
                cut,
                end=data.shape[0],
                with_corrected_st=True)
        spikes = extract_spikes(data, ep, mc=mc)
    else:
        if mc is True:
            size = 0, sum(cut), data.shape[1]
        else:
            size = 0, sum(cut) * data.shape[1]
        spikes = sp.zeros(size)

    # re-resample?
    if sample_back and rsf != 1.0:
        spikes = resample(spikes, spikes.shape[1] * 1. / rsf, axis=1)
        st *= 1. / rsf

    # return
    return spikes, st

##  MAIN

if __name__ == '__main__':
    pass
