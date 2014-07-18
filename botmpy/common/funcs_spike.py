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


"""functions for spike sorting"""
__docformat__ = 'restructuredtext'
__all__ = [
    'threshold_detection', 'merge_epochs', 'invert_epochs',
    'epochs_from_binvec', 'epochs_from_spiketrain',
    'epochs_from_spiketrain_set', 'chunk_data', 'extract_spikes',
    'get_cut', 'snr_maha', 'snr_peak', 'snr_power', 'overlaps']

##--- IMPORTS

import scipy as sp
from .util import *
from .funcs_general import sortrows

##---FUNCTIONS

# event detection

def threshold_detection(data, th, min_dist=1, mode='gt', find_max=True):
    """detect events by applying a threshold to the data

    :type data: ndarray
    :param data: the 2d-data to apply the threshold on. channels are in the
        second dimension (columns).
        Required
    :type th: ndarray or list
    :param th: list of threshold values, one value per channel in the `data`
        Required
    :type min_dist: int
    :param min_dist: minimal distance two successive events have to be
        separated in samples, else the event is ignored.
        Default=1
    :type mode: str
    :param mode: one of 'gt' for greater than or 'lt' for less than. will
        determine how the threshold is applied.
        Default='gt'
    :type find_max: bool
    :param find_max: if True, will find the maximum for each event epoch, else
        will find the start for each event epoch.
        Default=True
    :rtype: ndarray
    :returns: event samples
    """

    # checks
    data = sp.asarray(data)
    if data.ndim != 2:
        if data.ndim == 1:
            data = sp.atleast_2d(data).T
        else:
            raise ValueError('data.ndim != 2')
    th = sp.asarray(th)
    if th.ndim != 1:
        raise ValueError('th.ndim != 1')
    if th.size != data.shape[1]:
        raise ValueError('thresholds have to match the data channel count')
    if mode not in ['gt', 'lt']:
        raise ValueError('unknown mode, use one of \'lt\' or \'gt\'')
    if min_dist < 1:
        min_dist = 1

    # inits
    rval = []
    ep_func = {'gt': lambda d, t: epochs_from_binvec(d > t).tolist(),
               'lt': lambda d, t: epochs_from_binvec(d < t).tolist(),
    }[mode]

    # per channel detection
    for c in xrange(data.shape[1]):
        epochs = ep_func(data[:, c], th[c])
        if len(epochs) == 0:
            continue
        for e in xrange(len(epochs)):
            rval.append(epochs[e][0])
            if find_max is True:
                rval[-1] += data[epochs[e][0]:epochs[e][1] + 1, c].argmax()
    rval = sp.asarray(rval, dtype=INDEX_DTYPE)

    # do we have events?
    if rval.size == 0:
        return rval

    # drop event duplicates by sorting and checking for min_dist
    rval.sort()
    rval = rval[sp.diff(sp.concatenate(([0], rval))) >= min_dist]

    # return
    return rval

## epoch handling functions

def merge_epochs(*args, **kwargs):
    """for a set of epoch sets check if the combined set of epochs overlap
    and merge to one set with no overlapping epochs and no epochs of negative
    length.

    :param args: arbitrary count of epoch sets [[start, end]]
    :keyword min_dist: int - If present and greater than zero, this integer
        will be taken as the minimum distance in between epochs that is
        allowed. Should the gap in between two epochs smaller than min_dist,
        they are merged including the gap. This might reduce the
        segmentation of the data.
    :returns: ndarray - merged epoch set [[start, end]]
    """

    # checks
    for item in args:
        if not isinstance(item, (list, sp.ndarray)):
            raise ValueError('wrong inputs! lists and ndarrays allowed')

    # inits
    epochs = sortrows(sp.vstack(args)).tolist()
    if len(epochs) == 0:
        return sp.zeros((0, 2), dtype=INDEX_DTYPE)

    # rval_ovlp overlaps
    rval_ovlp = [epochs.pop(0)]
    k = 0
    min_dist = int(kwargs.get('min_dist', -1))

    while len(epochs) > 0:
        ep = epochs.pop(0)
        if ep[0] <= rval_ovlp[k][1] + min_dist:
            rval_ovlp[k] = [min(ep[0], rval_ovlp[k][0]),
                            max(ep[1], rval_ovlp[k][1])]
        else:
            k += 1
            rval_ovlp.append(ep)
    rval = rval_ovlp

    # return
    rval = sp.asarray(rval, dtype=INDEX_DTYPE)
    rval[rval[:, 0] < 0, :] = 0
    rval = rval[rval[:, 1] - rval[:, 0] > 0, :]
    return rval


def invert_epochs(epochs, end=None):
    """inverts epochs inverted

    The first epoch will be mapped to [0, start] and the last will be mapped
    to [end of last epoch, :end:]. Epochs that accidentally become negative
    or zero-length will be omitted.

    :type epochs: ndarray
    :param epochs: epoch set to invert
    :type end: int
    :param end: If not None, it i taken for the end of the last epoch,
        else max(index-dtype) is taken instead.
        Default=None
    :returns: ndarray - inverted epoch set
    """

    # checks
    if end is None:
        end = sp.iinfo(INDEX_DTYPE).max
    else:
        end = INDEX_DTYPE.type(end)

    # flip them
    rval = sp.vstack((
        sp.concatenate(([0], epochs[:, 1])),
        sp.concatenate((epochs[:, 0], [end])))).T
    return (rval[rval[:, 1] - rval[:, 0] > 0]).astype(INDEX_DTYPE)


def epochs_from_binvec(binvec):
    """returns the discrete epochs where the :binvec: is true

    :type binvec: ndarray
    :param binvec: one-domensinal boolean ndarray.
    :returns: ndarray - epoch set where :binvec: is True [[start, end]]
    """

    # early exit
    if not binvec.any():
        return sp.zeros((0, 2))

    # calculate
    output = sp.correlate(sp.concatenate(([0], binvec, [0])), [-1, 1], 'same')
    return sp.vstack((
        (output > 0).nonzero()[0] - 1,
        (output < 0).nonzero()[0] - 1)).T


def epochs_from_spiketrain(st, cut, end=None, with_corrected_st=False):
    """yields epoch set, given a spiketrain and cut parameter

    :type st: ndarray
    :param st: spiketrains as 1d array
    :type cut: tuple
    :param cut: 2-tuple of cutting parameters, (cut_left,cut_right) spike
        epochs will be generated by using cut_left and cut_right on the spike
        time. If an int is given, a symmetric cut tuple is assumed.
    :type end: int
    :param end: to determine potential problems with epochs overlapping data
        boundaries. If an event in the spiketrain is closer to 0 than :cut[0]:
        or closer to :end: than :cut[1]: the corresponding epoch will be
        omitted. If None, :end: will be set to max(INDEX_DTYPE)
        Default=None
    :type with_corrected_st: bool
    :param with_corrected_st: if True, return the corrected spiketrain by
        omitting spike events that cannot generate valid spike epochs given
        the passed cut settings.
        Default=False
    :returns: ndarray - epoch set of valid spike epochs, and if
        :with_corrected_st: is True additionally the corrected spike train
    """

    # checks
    st = sp.asarray(st)
    cut = get_cut(cut)
    if end is None:
        end = sp.iinfo(INDEX_DTYPE).max
    else:
        end = INDEX_DTYPE.type(end)

    # return the epochs for the spiketrain
    st_ok = (st >= cut[0]) * (st < end - cut[1])
    rval = sp.vstack((
        st[st_ok] - cut[0],
        st[st_ok] + cut[1])).T.astype(INDEX_DTYPE)
    ## FIX: astype is handling float entries weird sometimes! take care to pass spiketrains as integer arrays!
    ## we are now correcting spike epochs to be of length sum(cut) by pruning the start of the epoch
    tf = sum(cut)
    for i in xrange(rval.shape[0]):
        if rval[i, 1] - rval[i, 0] != tf:
            rval[i, 0] = rval[i, 1] - tf
    ## XIF
    if with_corrected_st is True:
        return rval, st[st_ok]
    else:
        return rval


def epochs_from_spiketrain_set(sts, cut, end=None):
    """yields epoch sets, given a spiketrain set and cut parameter

    one set for each unit plus one for the noise epochs in a dict

    :type sts: dict
    :param sts: dict with the spiketrains for each unit in the set. none of
        the units in the spiketrain set may have the key 'noise'!
    :type cut: tuple
    :param cut: 2-tuple of cutting parameters, (cut_left, cut_right) spike
        epochs will be generated by using cu_left and cut_right on the spike
        time. If an int is given, a symmetric cut tuple is assumed.
    :param end: to determine potential problems with epochs overlapping data
        boundaries. If an event in the spiketrain is closer to 0 than :cut[0]:
        or closer to :end: than :cut[1]: the corresponding epoch will be
        omitted. If None, :end: will be set to max(INDEX_DTYPE)
        Default=None
    :returns: dict - one epoch set per spike train plus the merged noise
        epoch set.
    """

    # inits and checks
    if not isinstance(sts, dict):
        raise ValueError('sts has to be a set of spiketrains in a dict')
    rval = {}

    # get the spiketrain epochs
    for key in sts:
        rval[key] = epochs_from_spiketrain(sts[key], cut, end=end)

    # add the noise epochs
    rval['noise'] = invert_epochs(merge_epochs(*rval.values()), end=end)

    # return the epoch set
    return rval

## spike and data extraction

def chunk_data(data, epochs=None, invert=False):
    """returns a generator of chunks from data given epochs

    :type data: ndarray
    :param data: signal data [[samples, channels]]
    :type epochs: ndarray
    :param epochs: epoch set, positive mask
    :type invert: bool
    :param invert: invert epochs, negative mask instead of positive mask
    :returns: generator - data chunks as per :epochs:
    """

    # checks
    data = sp.asarray(data)
    if data.ndim != 2:
        data = sp.atleast_2d(data).T
    if epochs is not None:
        if epochs.ndim != 2:
            raise ValueError('epochs has to be ndim=2 like [[start,end]]')
    if invert is True and epochs is not None:
        epochs = invert_epochs(epochs, end=data.shape[0])
    if epochs is None or len(epochs) == 0:
        epochs = [[0, data.shape[0]]]

    # yield data chunks
    for ep in epochs:
        yield data[ep[0]:ep[1], :], list(ep)


def extract_spikes(data, epochs, mc=False):
    """extract spike waveforms of size tf from data

    :type data: ndarray
    :param data: signal data [[samples, channels]]
    :type epochs: ndarray
    :param epochs: spike epoch set [[start,end]]
    :type mc: bool
    :param mc: if True, extract multichanneled spike waveforms as [n,tf,nc]
        else extract channel concatenated spike waveforms as [n, tf*nc]
        *False as default for legacy compatibility*
        Default=False
    :returns: ndarray - extracted spike data epochs
    """

    # checks
    data = sp.asarray(data)
    if data.ndim != 2:
        raise ValueError('data has to be ndim==2')
    if epochs.ndim != 2:
        raise ValueError('epochs has to be ndim==2')

    # inits
    nspikes = epochs.shape[0]
    if epochs.shape[0] == 0:
        # early exit
        return sp.zeros((0, epochs.shape[1]))
    tf, nc = epochs[0, 1] - epochs[0, 0], data.shape[1]

    # extract
    if mc is True:
        rval = sp.zeros((nspikes, tf, nc), dtype=data.dtype)
    else:
        rval = sp.zeros((nspikes, tf * nc), dtype=data.dtype)
    for s in xrange(nspikes):
        for c in xrange(nc):
            clamp = 0
            if epochs[s, 1] > data.shape[0]:
                clamp = epochs[s, 1] - data.shape[0]
            if mc is True:
                rval[s, :tf - clamp, c] = \
                    data[epochs[s, 0]:epochs[s, 1] - clamp, c]
            else:
                rval[s, c * tf:(c + 1) * tf - clamp] = \
                    data[epochs[s, 0]:epochs[s, 1] - clamp, c]

    # return
    return rval


def get_cut(tf, off=0):
    """cut 2-tuple (cut_left,cut_right) generating function

    Used to generate epochs from events. Per default the epoch will be
    placed symmetrically around the event sample. :off: can be used to
    influence the placement. For odd tf values the extension of the
    cut_right part will be favored.

    :type tf: int
    :param tf: length of the waveform in samples
    :type off: int
    :param off: offset for epoch start/end
        Default=0
    """
    if isinstance(tf, tuple):
        if len(tf) == 2:
            return tf[0] - int(off), tf[1] + int(off)
        else:
            raise ValueError('tuples have to be of length==2 for get_cut')
    elif isinstance(tf, int):
        return int(tf / 2.0) - int(off), int(tf / 2.0) + tf % 2 + int(off)
    else:
        raise TypeError('only int or tuple are allowed for get_cut')

## SNR functions - added by Felix 10. aug 2009

def snr_peak(waveforms, noise_var):
    """SNR from instantaneous variance

    Definition of signal to noise ratio (SNR) as the ratio between the peak of
    a waveforms and the noise standard deviation.

    :type waveforms: ndarray
    :param waveforms: waveform data (signal), one per row
    :type noise_var: float
    :param noise_var: instantaneous variance of the noise (noise)
    :returns: ndarray - SNR per waveform
    """

    return sp.absolute(waveforms).max(axis=1) / sp.sqrt(noise_var)


def snr_power(waveforms, noise_var):
    """SNR from signal energy

    Definition of signal to noise ratio (SNR) using the waveforms energy as
    defined by Rutishauser et al (2006)

    :type waveforms: ndarray
    :param waveforms: waveform data (signal), one per row
    :type noise_var: float
    :param noise_var: instantaneous variance of the noise (noise)
    :returns: ndarray - SNR per waveform
    """

    denom = waveforms.shape[1] * noise_var
    return sp.sqrt((waveforms * waveforms).sum(axis=1) / denom)


def snr_maha(waveforms, invC, mu=None):
    """SNR from Mahalanobis distance (generalised euclidean distance)

    Definition of signal to noise ratio (SNR) as derived from the Mahalanobis
    distance. For C=eye this is equivalent to snr_power.

    :type waveforms: ndarray
    :param waveforms: waveform data (signal), one per row
    :type invC: ndarray
    :param invC: inverted noise covariance matrix (a block toeplitz matrix)
    :type mu: ndarray
    :param mu: mean correction. Usually we assume zero-mean waveforms,
        so if this is None it will be ignored.
        Default=None
    :returns: ndarray - SNR per waveform
    """

    # inits and checks
    n, dim = waveforms.shape
    if dim != invC.shape[0] or dim != invC.shape[1]:
        raise ValueError('dimension mismatch for waveforms and covariance')
    rval = sp.zeros(n)

    # correct for mu
    if mu is not None:
        if mu.shape != (dim,):
            raise ValueError('dimension mismatch for waveforms and mu')
        waveforms -= mu

    # compute
    for i in xrange(n):
        rval[i] = sp.dot(sp.dot(waveforms[i], invC), waveforms[i].T)
        rval[i] /= float(dim)
    return sp.sqrt(rval)

## data processing algorithms

def overlaps(sts, window):
    """produces dict of boolean sequences indicating for all spikes in all
    spike trains in :sts: if it participates in an overlap event.

    :type sts: dict
    :param sts: spike train set
    :type window: int
    :param window: overlap window size
    :returns: dict - boolean spike train set
    """

    # inits
    n = len(sts)
    ovlp, ovlp_nums = {}, {}
    for k in sts.keys():
        ovlp[k] = sp.zeros(sts[k].shape, dtype=bool)
        ovlp_nums[k] = 0

    # run over all pairs of spike trains in sts
    for i in xrange(n):
        i_ = sts.keys()[i]
        trainI = sts[i_]
        for j in xrange(i + 1, n):
            # for every pair run over all spikes in i and check if a spike
            # in j overlaps
            j_ = sts.keys()[j]
            trainJ = sts[j_]

            for spkI, spk in enumerate(trainI):
                d = trainJ - spk
                overlap_indices = sp.absolute(d) < window
                if not overlap_indices.sum():
                    continue
                ovlp_nums[i_] += 1
                ovlp_nums[j_] = overlap_indices.sum()

                ovlp[i_][spkI] = True
                ovlp[j_][overlap_indices] = True

    return ovlp, ovlp_nums

##--- MAIN

if __name__ == '__main__':
    pass
