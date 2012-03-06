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

These detecors find events and event epochs on potentially multichanneled data
signal. Mostly, you will want to reset the internals of the detector after
processing a chunk of data. There are different kinds of detectors, the common
product of the detector is the discrete events or epochs in the data signal.

DATA_TYPE: fixed for float32 (single precision)"""

__docformat__ = 'restructuredtext'
__all__ = ['ArtifactDetectorNode']

##--- IMPORTS

import scipy as sp
from ..common import (epochs_from_binvec, merge_epochs, invert_epochs,
                      INDEX_DTYPE)
from .detector_nodes import ThresholdDetectorNode

##--- CLASSES

class ArtifactDetectorNode(ThresholdDetectorNode):
    """detects artifacts by detecting zero-crossing frequencies

    For a zero-mean gaussian process the the zero-crossing rate zcr is
    independent of the moments and its expectation is approaches 0.5 as the
    integration-window size approaches infinity.

    The capacitive artifacts seen in the Munk dataset have a significantly
    lower
    frequency, such that the zcr decreases to 0.1 and below. Detecting the zcr
    and blocking epochs where the zcr significantly deviates from the
    expectation imposed by the gaussian noise process can lead to detection of
    artifact epochs.

    The zero crossing rate (zcr) is given by the convolution of a moving
    average
    window (although this is configurable to use other weighting methods) with
    the XOR of the signbits of X(t) and X(t+1).
    """

    ## constructor

    def __init__(self, wsize_ms=15.0, psize_ms=(5.0, 10.0), wfunc=sp.ones,
                 srate=32000.0, zcr_th=0.1, mindist_ms=10.0):
        """
        :Parameters:
            wsize_ms : float
                Window size in ms for the integration window. This should be
                large enough to cover the low band of the artifacts.
                Default=15.0
            psize_ms : float
                Padding size in ms. Detected epochs will be extended by the
                padding window on both ends, to limit unnecessary segmentation
                of the data.
                Default=5.0
            wfunc : func
                Function that creates the integration window. The function has
                to take one parameter denoting the window size in samples.
                Default=scipy.ones
            srate : float
                The sample rate in Hz. Used to convert the windows sizes in ms
                representation into their sampled representation.
                Default=32000.0
            zcr_th : float
                The zrc (zero crossing rate) threshold, epochs where the zrc
                falls below the threshold will be classified as artifact
                epochs.
                Default=0.11
            mindist_ms : float
                The minimum size for non-artifact epochs. Epochs in between
                artifacts that are smaller than this window in ms, are merged
                into the artifact epochs.
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
        self.threshold = float(zcr_th)

    ## privates

    def _energy_func(self, x, **kwargs):
        x_signs = sp.signbit(x)
        return sp.vstack((
            sp.bitwise_xor(x_signs[:-1], x_signs[1:]),
            [False] * x.shape[1]
            ))

    def _execute(self, x, *args, **kwargs):
        # inits
        epochs = []

        # per channel detection
        for c in xrange(self.nchan):
            # filter energy with window
            xings = sp.correlate(self.energy[:, c], self.window, 'same')
            # replace range at start and end of signal with the mean of the
            # rest
            mu = xings[self.window.size:-self.window.size].mean()
            xings[:self.window.size] = xings[-self.window.size:] = mu
            ep = epochs_from_binvec(xings < self.threshold)
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

##--- MAIN

if __name__ == '__main__':
    from os import listdir, path as osp
    from spikeplot import mcdata, plt
    from spikepy.common import XpdFile
    from spikepy.nodes import SDMteoNode as SDET

    tf = 65
    AD = ArtifactDetectorNode()
    SD = SDET(tf=tf, min_dist=int(tf * 0.5))
    XPDPATH = '/home/phil/Data/Munk/Louis/L011'

    for fname in sorted(filter(lambda x:x.startswith('L011') and
                                        x.endswith('.xpd'),
                               listdir(XPDPATH)))[
                 :20]:
        arc = XpdFile(osp.join(XPDPATH, fname))
        data = arc.get_data(item=7)
        AD(data)
        print AD.events
        print AD.get_nonartefact_epochs()
        print AD.get_fragmentation()
        SD(data)
        f = mcdata(data=data, other=SD.energy, events={0:SD.events},
                   epochs=AD.events, show=False)
        for t in SD.threshold:
            f.axes[-1].axhline(t)
        plt.show()
