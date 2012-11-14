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


"""detector nodes for capacitative artifacts in multichanneled data

These detectors find events and event epochs on potentially multichanneled data
signal. Mostly, you will want to reset the internals of the detector after
processing a chunk of data. There are different kinds of detectors, the common
product of the detector is the discrete events or epochs in the data signal.

DATA_TYPE: fixed for float32 (single precision)"""

__docformat__ = 'restructuredtext'
__all__ = ['ArtifactDetectorNode', 'SpectrumArtifactDetector']

##--- IMPORTS

import scipy as sp
from ..common import epochs_from_binvec, merge_epochs, invert_epochs, INDEX_DTYPE
from .spike_detection import ThresholdDetectorNode

##--- CLASSES

class ArtifactDetectorNode(ThresholdDetectorNode):
    """detects artifacts by detecting zero-crossing frequencies

    For a zero-mean gaussian process the the zero-crossing rate `zcr` is
    independent of its moments and approaches 0.5 as the integration window
    size approaches infinity:

    .. math::

        s_t \\sim N(0,\\Sigma)

        zcr_{wsize}(s_t) = \\frac{1}{wsize-1} \\sum_{t=1}^{wsize-1}
        {{\\mathbb I}\\left\{{s_t s_{t-1} < 0}\\right\\}}

        \\lim_{wsize \\rightarrow \\infty} zcr_{wsize}(s_t) = 0.5

    The capacitive artifacts seen in the Munk dataset have a significantly
    lower frequency, s.t. zcr decreases to 0.1 and below, for the integration
    window sizes relevant to our application. Detecting epochs where the zcr
    significantly deviates from the expectation, assuming a coloured Gaussian
    noise process, can thus lead be used for detection of artifact epochs.

    The zero crossing rate (zcr) is given by the convolution of a moving
    average window (although this is configurable to use other weighting
    methods) with the XOR of the signbits of X(t) and X(t+1).
    """

    ## constructor

    def __init__(self, wsize_ms=15.0, psize_ms=(5.0, 10.0), wfunc=sp.ones,
                 srate=32000.0, zcr_th=0.1, mindist_ms=10.0):
        """
        :type wsize_ms: float
        :param wsize_ms: window size of the integration window in `ms`. Should
            be large enough to cover the low band of the artifacts and not
            overlap with the lower band of spikes (spike clusters).
            Default=15.0
        :type psize_ms: tuple
        :param psize_ms: window size of the padding windows in `ms`. Will be
            applied to detected artifact epochs. (left_pad, right_pad)
            Default=5.0
        :type wfunc: function
        :param wfunc: function that creates the integration window. The
            function has to take one parameter denoting the window size in
            samples.
            Default=scipy.ones
        :type srate: float
        :param srate: sample rate in `Hz`. Used to convert the windows sizes
            from `ms` to data samples.
            Default=32000.0
        :type zcr_th: float
        :param zrc_th: zrc (zero crossing rate) threshold, epochs of the data
            where the zrc falls below the threshold will be classified as
            artifact epochs.
            Default=0.11
        :type mindist_ms: float
        :param mindist_ms: minimum size for non-artifact epochs in `ms`.
            Data epochs in between artifacts epochs that are smaller than this
            window, are merged into the artifact epochs to reduce
            segmentation.
            Default=10.0
        """

        # super
        super(ArtifactDetectorNode, self).__init__()

        # members
        self.srate = float(srate)
        self.window = wfunc(int(wsize_ms * self.srate / 1000.0))
        self.window /= self.window.sum()
        self.pad = (int(psize_ms[0] * self.srate / 1000.0),
                    int(psize_ms[1] * self.srate / 1000.0))
        self.mindist = int(mindist_ms * self.srate / 1000.0)
        self.zcr_th = float(zcr_th)

    ## privates

    def _energy_func(self, x, **kwargs):
        x_signs = sp.signbit(x)
        return sp.vstack((sp.bitwise_xor(x_signs[:-1], x_signs[1:]), [False] * x.shape[1]))

    def _execute(self, x, *args, **kwargs):
        # init
        epochs = []

        # per channel detection
        for c in xrange(self.nchan):
            # filter energy with window
            xings = sp.correlate(self.energy[:, c], self.window, 'same')
            # replace filter artifacts with the mean
            mu = xings[self.window.size:-self.window.size].mean()
            xings[:self.window.size] = xings[-self.window.size:] = mu
            ep = epochs_from_binvec(xings < self.zcr_th)
            epochs.append(ep)

        # pad and merge artifact epochs
        epochs = sp.vstack(epochs)
        if epochs.size > 0:
            epochs[:, 0] -= self.pad[0]
            epochs[:, 1] += self.pad[1]
        self.events = merge_epochs(epochs, min_dist=self.mindist)

        # return
        self.events = self.events.astype(INDEX_DTYPE)
        return x

    ## evaluations

    def get_fragmentation(self):
        """returns the artifact fragmentation"""

        if self.size is None:
            raise RuntimeError('No data given!')
        nae_len = float(
            self.size - (self.events[:, 1] - self.events[:, 0]).sum())
        return - sp.log(nae_len / (self.size * (self.events.shape[0] + 1)))

    def get_nonartefact_epochs(self):
        """return the index set that represents the non-artifact epochs"""

        if self.size is None:
            raise RuntimeError('No data given!')
        if self.events.size == 0:
            return sp.array([[0, self.size]])
        else:
            return invert_epochs(self.events, end=self.size)


class SpectrumArtifactDetector(ThresholdDetectorNode):
    """detects artifacts by identifying unwanted frequency packages in the spectrum of the signal

            For a zero-mean gaussian process the the zero-crossing rate `zcr` is
            independent of its moments and approaches 0.5 as the integration window
            size approaches infinity:

            .. math::

                s_t \\sim N(0,\\Sigma)

                zcr_{wsize}(s_t) = \\frac{1}{wsize-1} \\sum_{t=1}^{wsize-1}
                {{\\mathbb I}\\left\{{s_t s_{t-1} < 0}\\right\\}}

                \\lim_{wsize \\rightarrow \\infty} zcr_{wsize}(s_t) = 0.5

            The capacitive artifacts seen in the Munk dataset have a significantly
            lower frequency, s.t. zcr decreases to 0.1 and below, for the integration
            window sizes relevant to our application. Detecting epochs where the zcr
            significantly deviates from the expectation, assuming a coloured Gaussian
            noise process, can thus lead be used for detection of artifact epochs.

            The zero crossing rate (zcr) is given by the convolution of a moving
            average window (although this is configurable to use other weighting
            methods) with the XOR of the signbits of X(t) and X(t+1).
            """

    ## constructor

    def __init__(self, wsize_ms=8.0, srate=32000.0, cutoff_hz=1000.0, nfft=256, **kw):
        """lala"""

        # super
        kw['ch_separate'] = True
        super(SpectrumArtifactDetector, self).__init__(**kw)

        # members
        self.srate = float(srate)
        self.wsize = None
        self.cutoff_hz = float(cutoff_hz)
        self.nfft = 1
        while self.nfft < nfft:
            self.nfft <<= 1

    ## privates

    def _threshold_func(self, x):
        return 1.0

    def _energy_func(self, x, **kwargs):
        from matplotlib.mlab import specgram

        rval = sp.zeros_like(x)
        ns, nc = x.shape
        for c in xrange(nc):
            psd_arr, freqs, times = specgram(x[:, c], NFFT=self.nfft, Fs=self.srate, noverlap=self.nfft / 2)
            mask = freqs < self.cutoff_hz
            for b in xrange(len(times)):
                bin_s = int(times[0] * b * self.srate) - self.nfft / 2
                bin_e = int(times[0] * b * self.srate) + self.nfft / 2
                rval[bin_s:bin_e, c] = psd_arr[mask == True, b].sum() / psd_arr[mask == False, b].sum()
                #rval[bin_s:bin_e, c] = psd_arr[mask == True, b].max() / psd_arr[mask == False, b].max()
        return rval

    def _execute(self, x, *args, **kwargs):
        # init
        epochs = []
        self._calc_threshold()

        # per channel detection
        for c in xrange(self.nchan):
            ep = epochs_from_binvec(self.energy[:, c] > self.threshold[c])
            epochs.extend(ep)
        if len(epochs) == 0:
            epochs = sp.zeros((0, 2))
        else:
            epochs = merge_epochs(epochs, min_dist=self.nfft + 1)
            epochs = epochs[epochs[:, 1] - epochs[:, 0] > self.nfft * 2]
        self.events = sp.asarray(epochs, dtype=INDEX_DTYPE)
        return x

    ## publics

    def get_nonartefact_epochs(self):
        """return the index set that represents the non-artifact epochs"""

        if self.size is None:
            raise RuntimeError('No data given!')
        if self.events.size == 0:
            return sp.array([[0, self.size]])
        else:
            return invert_epochs(self.events, end=self.size)

##--- MAIN

if __name__ == '__main__':
    pass
