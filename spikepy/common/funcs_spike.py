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


"""functions for spike sorting"""
__docformat__ = 'restructuredtext'
__all__ = ['threshold_detection', 'epochs_from_binvec',
           'epochs_from_spiketrain', 'epochs_from_spiketrain_set',
           'extract_spikes', 'get_cut', 'chunk_data', 'invert_epochs',
           'merge_epochs', 'prewhiten', 'snr_maha', 'snr_peak', 'snr_power',
           'overlaps']

##--- IMPORTS

from scipy import linalg as sp_la
from spikepy.common.funcs_general import sortrows
from spikepy.common.constants import *

##---FUNCTIONS

# event detection

def threshold_detection(data, th, min_dist=1, mode='gt', find_max=True):
    """detect events by applying a threshold to the data

    :Parameters:
        data : ndarray
            the 2d-data to apply the theshold on. channels are in the second
            dimension (columns).
            Required
        th : ndarray or list
            list of threshold values, one value per channel in the data
            Required
        min_dist : int
            minimal distance two successive events have to be separated in
            samples, else the event is ignored.
            Default=1
        mode : str
            one of 'gt' for greater than or 'lt' for less than. will determine
            how the threshold is applied.
            Default='gt'
        find_max : bool
            if True, will find the maximum for each event epoch,
            else will find
            the start for each event epoch.
            Default=True
    """
    # checks
    if data.ndim != 2:
        if data.ndim == 1:
            data = sp.atleast_2d(data).T
        else:
            raise ValueError('data has to be like  1 <= ndim <= 2')
    if th.ndim != 1:
        raise ValueError('thresholds have to be 1d!')
    if th.size != data.shape[1]:
        raise ValueError('thresholds have to match the data channel count')
    if mode not in ['gt', 'lt']:
        raise ValueError('unknown mode, use one of \'lt\' or \'gt\'')
    if min_dist < 1:
        min_dist = 1

    # inits
    rval = []
    ep_func = {
        'gt':lambda d, t:epochs_from_binvec(d > t).tolist(),
        'lt':lambda d, t:epochs_from_binvec(d < t).tolist(),
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
    """check for a set of epochs if they overlap and rval_ovlp to one
    consistent set

    Consistency means no overlapping epochs and no epoch of negative length.

    :Parameters:
        args : tuple
            arbitrary count of [[start, end]] epochs sets
        kwargs : dict
            keywords
    :Keywords:
        'min_dist' : int
            If present and greater than zero, this integer will be taken as
            the
            minimum distance in between epochs that is allowed. Should the gap
            in between two epochs smaller than min_dist, they are merged
            including the gap. This might reduce the segmentation of the data.
    :Returns:
        ndarray : merged and consitent set of epochs [[start, end]]
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
    while len(epochs) > 0:
        ep = epochs.pop(0)
        if ep[0] <= rval_ovlp[k][1] - 1:
            rval_ovlp[k] = [min(ep[0], rval_ovlp[k][0]),
                            max(ep[1], rval_ovlp[k][1])]
        else:
            k += 1
            rval_ovlp.append(ep)
    rval = rval_ovlp

    # rval_ovlp epochs with gaps smaller than minimum distance
    if 'min_dist' in kwargs:
        min_dist = int(kwargs.get('min_dist', 0))
        if min_dist > 0:
            rval_gaps = [rval_ovlp.pop(0)]
            while len(rval_ovlp) > 0:
                ep = rval_ovlp.pop(0)
                if ep[0] - rval_gaps[-1][1] < min_dist:
                    rval_gaps[-1][1] = ep[1]
                else:
                    rval_gaps.append(ep)
            rval = rval_gaps

    # return
    rval = sp.asarray(rval, dtype=INDEX_DTYPE)
    rval[rval[:, 0] < 0, :] = 0
    rval = rval[rval[:, 1] - rval[:, 0] > 0, :]
    return rval


def invert_epochs(epochs, end=None):
    """inverts epochs inverted

    The first epoch will be mapped to [0, start] and the last will be mapped
     to
    [start next of last, start of last] or

    :Parameters:
        epochs : ndarray
            epochs to invert
        end : int
            If not None and an integer value, it i taken for the end point of
            the last epochs, if None max(index-dtype) is taken instead.
    :Returns:
        ndarray
            inverted epochs
    """

    # inits
    if end is None:
        end = sp.iinfo(INDEX_DTYPE).max
    else:
        end = end

    # flip them
    rval = sp.vstack((
        sp.concatenate(([0], epochs[:, 1])),
        sp.concatenate((epochs[:, 0], [end]))
        )).T
    return (rval[rval[:, 1] - rval[:, 0] > 0]).astype(INDEX_DTYPE)


def epochs_from_binvec(binvec):
    """returns the epochs where the passed binary vector is true

    :Parameters:
        binvec : ndarray
            A 1d-boolean-ndarray.
    :Returns:
        ndarray
            Epochs where binvec is True. [[start, end]]
    """

    # early exit
    if not binvec.any():
        return sp.zeros((0, 2))

    # calculate
    output = sp.correlate(sp.concatenate(([0], binvec, [0])), [-1, 1], 'same')
    return sp.vstack((
        (output > 0).nonzero()[0] - 1,
        (output < 0).nonzero()[0] - 2
        )).T


def epochs_from_spiketrain(st, cut, end=None, with_corrected_st=False):
    """yields epochs, given a spiketrain and cut parameters

    :Parameters:
        st : ndarray
            the spiketrains as 1d array
        cut : tuple
            2-tuple of cutting parameters, (cut_left, cut_right)
            spikes-epochs will be generated by using cut_left and cut_right
            on the spike time. If an int is given, a symmetric cut tuple is
            assumed.
        end : int
            To determine potential problems with epochs overlapping data
            boundaries. If an event in the spiketrain is closer to end that
            the
            cut value, the corresponding epoch will be droped. If end is None,
            no such checking will be done.
    """

    # inits and checks
    if not isinstance(st, sp.ndarray):
        raise ValueError('st has to be a ndarray')
    if not isinstance(cut, (int, tuple)):
        raise ValueError('cut has to be either a 2-tuple or an int')
    else:
        if isinstance(cut, int):
            cut = get_cut(cut)

    # return the epochs for the spiketrain
    st_ok = (st >= cut[0])
    if end is not None:
        st_ok *= (st < end - cut[1])
    rval = sp.vstack((
        st[st_ok] - cut[0],
        st[st_ok] + cut[1]
        )).T.astype(INDEX_DTYPE)
    if with_corrected_st is True:
        return rval, st[st_ok]
    else:
        return rval


def epochs_from_spiketrain_set(sts, cut, end=None):
    """yields epoch sets, given a spiketrain set and cut parameters

    one set for each unit plus one for the noise epochs

    :Parameters:
        sts : dict
            dictionary with the spike trains for each unit in the set.
            none of the units in the spiketrain set may have the key 'noise'!
        cut : tuple
            2-tuple of cutting parameters, (cut_left, cut_right)
            spikes-epochs will be generated by using cu_left and cut_right
            on the spike time. If an int is given, a symmetric cut tuple is
            assumed.
        end : int
            sample of the end of the data window to get the end of the last
            noise epoch right. If None, this will default to max(INDEX_DTYPE)
            Default=None
    """

    # inits and checks
    if not isinstance(sts, dict):
        raise ValueError('sts has to be a set of spiketrains in a dict')
    if not isinstance(cut, (int, tuple)):
        raise ValueError('cut has to be either a 2-tuple or an int')
    else:
        if isinstance(cut, int):
            cut = get_cut(cut)
    if end is None:
        end = sp.iinfo(INDEX_DTYPE).max
    else:
        end = end
    rval = {}

    # get the spiketrain epochs
    for key in sts:
        rval[key] = epochs_from_spiketrain(sts[key], cut, end=end)

    # add the noise epochs
    rval['noise'] = invert_epochs(merge_epochs(*rval.values()), end=end)

    # return the epoch set
    return rval

## spike and data extraction

def chunk_data(data, epochs=None):
    """returns a generator of chunks from data given epochs"""

    # checks and inits
    if not isinstance(data, sp.ndarray):
        raise ValueError('data is not an ndarray')
    if data.ndim != 2:
        raise ValueError('data has be ndim==2')
    if epochs is None or len(epochs) < 1:
        epochs = [[0, data.shape[0]]]

    # generate data chunks
    for ep in epochs:
        if len(ep) != 2:
            raise ValueError('invalid epoch: %s' % ep)
        yield data[ep[0]:ep[1], :], list(ep)


def extract_spikes(data, epochs, mc=False):
    """extract spike waveforms of size tf from data

    :Parameters:
        data : ndarray
            the signal. channels on the columns
        epochs : ndarray
            epochs [[start,end]]
        mc : bool
            if True, extract multichanneled spikes as [n, tf, nc] else extract
            concatenated spikes as [n, tf*nc]
            *False as default for legacy compatibility*
            Default=False
    :Returns:
        ndarray
            Extracted epochs from data as either [m,tf,nc] or [n, tf*nc].

    """

    # inits and checks
    nspikes = epochs.shape[0]
    if epochs.shape[0] == 0:
        return sp.zeros((0, epochs.shape[1]))
    tf = epochs[0, 1] - epochs[0, 0]
    nc = data.shape[1]

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
                rval[s, :tf - clamp, c] = data[
                                          epochs[s, 0]:epochs[s, 1] - clamp,
                                          c]
            else:
                rval[s, c * tf:(c + 1) * tf - clamp] = data[
                                                       epochs[s, 0]:epochs[
                                                                    s,
                                                                    1] - clamp
                , c]

    # return
    return rval


def get_cut(tf, off=0):
    """returns the tuple for cutting out waveforms

    :Parameters:
        tf : int
            length of the waveform in samples
            Required
        off : int
            offset for cutting out
            Default=0
    """

    return int(tf / 2.0) - int(off), int(tf / 2.0) + tf % 2 + int(off)

## SNR functions - added by Felix 10. aug 2009

def snr_peak(data, noise_var):
    """standard SNR

    Standard definition of signal to noise ratio (SNR) as the ratio between
    the
    peak of a waveform and the noise standard deviation.

    :Parameters:
        data : ndarray
            the signal
        noise_var : float
            the noise variance
    :Returns:
        float
            the SNR
    """

    return sp.absolute(data).max() / sp.sqrt(noise_var)


def snr_power(data, noise_var):
    """energy based SNR

    Signal to noise ratio as defined by Rutishauser et al (2006)
    (only for single channel waveforms)

    :Parameters:
        data : ndarray
            the signal
        noise_var : float
            the noise variance
    :Returns:
        ndarray
            the SNR for each channel
    """

    dim = data.shape[1]
    return sp.sqrt((data * data).sum(1) / (dim * noise_var))


def snr_maha(data, invC, mu=None):
    """mahalanobis distance SNR

    Signal to noise ratio as derived from the Mahalanobis Distance. Only for
    single channel waveforms. This is in the case of C being the identity
    matrix
    the same as snr_power

    :Parameters:
        data : ndarray
            the templates, concatenated across the channels, one per row
        invC : ndarray
            noise covariance marix as a block toeplitz matrix
        mu : ndarray
            Mean correction. Usually we assume zero-mean data.
    :Returns:
        ndarray
            the SNR for each row of data, w.r.t invC
    """

    # inits and checks
    n, dim = data.shape
    if dim != invC.shape[0] or dim != invC.shape[1]:
        raise ValueError('dimension mismatch for data and covariance')
    rval = sp.zeros(n, dtype=data.dtype)

    # correct for mu
    if mu is not None:
        if mu.shape != (dim,):
            raise ValueError('dimension mismatch for data and mu')
        mydata = data - mu
    else:
        mydata = data

    # compute
    for i in xrange(n):
        rval[i] = sp.sqrt(
            sp.dot(sp.dot(mydata[i, :], invC), mydata[i, :].T) / dim
        )

    # fire away
    return rval

## data processing algorithms

def prewhiten(data, ncov):
    """prewhiten data with respect to a noise covariance matrix"""

    # inits and checks
    if isinstance(data, sp.ndarray):
        data = {0:data}
    size = data[sorted(data.keys())[0]].shape[1]
    dtype = data[sorted(data.keys())[0]].dtype
    if ncov is None:
        ncov = sp.eye(size, dtype=dtype)
        ncov += 0.01 * sp.randn(*ncov.shape)

    # factorize
    try:
        cholncov = sp_la.cholesky(ncov)
    except:
        ncov += 0.0001 * sp.randn(*ncov.shape)
        cholncov = sp_la.cholesky(ncov)
    iU = sp_la.inv(cholncov)

    # apply to data
    pwdata = {}
    for k in sorted(data.keys()):
        if data[k].shape[1] != ncov.shape[1]:
            raise ValueError(
                'dimension mismatch for data (%d) and covariance (%d)' % (
                    data[k].shape[1], ncov.shape[1]))
        pwdata[k] = sp.dot(data[k], iU).astype(dtype)

    # return
    return pwdata, iU, cholncov


def overlaps(G, window):
    """Calculates a "boolean" dict, indicating for every spike in every spike
    train in G whether it belongs to an overlap or not.
    """

    n = len(G)
    O = {}
    for k in G.keys():
        O[k] = sp.zeros(G[k].shape, dtype=sp.bool_)
    Onums = sp.zeros(len(G))
    # run over all pairs of spike trains in G
    for i in xrange(n):
        for j in xrange(i + 1, n):
            # for every pair run over all spikes in i and check whether a spike
            # in j overlaps
            trainI = G[G.keys()[i]]
            trainJ = G[G.keys()[j]]
            idxI = 0
            idxJ = 0
            while idxI < len(trainI) and idxJ < len(trainJ):
                # Overlapping?
                if abs(trainI[idxI] - trainJ[idxJ]) < window:
                    # Every spike can only be in one or no overlap. prevents triple
                    # counting
                    if O[G.keys()[i]][idxI] == 0:
                        O[G.keys()[i]][idxI] = 1
                        Onums[i] += 1
                    if O[G.keys()[j]][idxJ] == 0:
                        O[G.keys()[j]][idxJ] = 1
                        Onums[j] += 1

                if trainI[idxI] < trainJ[idxJ]:
                    idxI += 1
                else:
                    idxJ += 1
    ret = {'O':O, 'Onums':Onums}
    return ret

##--- MAIN

if __name__ == '__main__':
    pass
