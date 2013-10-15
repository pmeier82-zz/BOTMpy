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

"""detector nodes for multichanneled data

These detecors find features and feature epochs on multichanneled signals.
Mostly, you will want to reset the internals of the detector after
processing a
chunk of data, which is featured by deriving from ResetNode. There are
different
kinds of detectors, distinguished by their way of feature to noise
discrimination.
"""

__docformat__ = 'restructuredtext'
__all__ = ['EnergyNotCalculatedError', 'ThresholdDetectorNode', 'SDAbsNode',
           'SDSqrNode', 'SDMteoNode', 'SDKteoNode', 'SDIntraNode', 'SDPeakNode']

##--- IMPORTS

import scipy as sp
from scipy.stats.mstats import mquantiles
from .base_nodes import ResetNode
from ..common import (threshold_detection, extract_spikes, merge_epochs,
                      get_cut, kteo, mteo, INDEX_DTYPE, get_aligned_spikes)

##--- CLASSES

class EnergyNotCalculatedError(Exception):
    """EnergyNotCalculatedError Exception"""

    def __init__(self):
        super(EnergyNotCalculatedError, self).__init__('self.energy is None')


class ThresholdDetectorNode(ResetNode):
    """abstract interface for detecting feature epochs in a signal

    The ThresholdDetectorNode is the abstract interface for all detectors. The
    input signal is assumed to be a (multi-channeled) signal,
    with data for one
    channel in each column (or one multi-channeled observation/sample per
    row).

    The output will be a timeseries of detected feature in the input signal.
    To find the features, the input signal is transformed by applying an
    operator
    (called the energy function from here on) that produces an
    representation of the input signal, which should optimize the SNR of the
    features vs the
    remainder of the input signal. A threshold is then applied to this energy
    representation of the input signal to find the feature epochs.

    The output timeseries either holds the onsets of the feature epochs or the
    maximum of the energy function within the feature epoch, givin in samples.

    Extra information about the events or the internals has to be saved in
    member variables along with a proper interface.
    """

    ## constructor

    def __init__(self, input_dim=None, output_dim=None, dtype=None,
                 energy_func=None, threshold_func=None, threshold_mode='gt',
                 threshold_base='energy', threshold_factor=1.0, tf=47,
                 min_dist=1, find_max=True, ch_separate=False):
        """
        see mdp.Node
        :type energy_func: function
        :param energy_func: function handle to calculate the energy of the
            input signal. If this parameter is specified, self._energy_func
            will be replaced with the function passed. The energy operator
            should take the input signal as the only input argument.
            Default=None
        :type threshold_func: function
        :param threshold_func: function handle to calculate the thresholds for
            feature epoch detection. The threshold function has to return a
            scalar value and will be used as the threshold. If this parameter
            is specified, self._threshold_func will be replaced with the
            function passed.
            Default=None
        :type threshold_mode: str
        :param threshold_mode: one of 'gt' or 'lt'. Defines wether the
            threshold is applied with the 'gt' (greater than) or 'lt' (less
            than) mode.
            Default='gt'
        :type threshold_base: str
        :param threshold_base: one of 'signal' or 'energy'. Determines what
            quantity is taken to derive the threshold from. If 'signal', the
            current input signal will be taken to derive the threshold from.
            If 'energy', the energy representation of current input signal
            will be taken to derive the threshold from.
            Default='energy'
        :type threshold_factor: float
        :param threshold_factor: Scalar to adjust the threshold linearly.
            Threshold will be set at threshold_factor * threshold_func
            (threhold_base).
        :type tf: int
        :param tf: The width/length in samples ot the features to be detected.
            Used for extraction and self.get_epochs.
            Default=47
        :type min_dist: int
        :param min:dist: Minimum distance in samples that has to lie in
            between two detected feature epochs, so they will be detected as
            two distinct feature epochs. Feature epochs closer than min_dist
            will be merged into one feature epoch.
            Default=1
        :type find_max: bool
        :param find_max: If True, will find feature as the maxima in the
            feature epoch. Else, the onset of feature epoch will be taken as
            the event.
            Default=True
        :type ch_separate: bool
        :param ch_separate: if True, find event per channel separatly, else
            use the max along the signal energy function.
            Default=False
        """

        # super
        super(ThresholdDetectorNode, self).__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            dtype=dtype)

        # members
        self.tf = int(tf)
        self.min_dist = int(min_dist)
        self.find_max = bool(find_max)
        if threshold_mode not in ['gt', 'lt']:
            raise ValueError('threshold mode must be either "gt" or "lt"')
        self.th_mode = threshold_mode
        if threshold_base not in ['signal', 'energy']:
            raise ValueError(
                'threshold base must be either "signal" or "energy"')
        self.th_base = threshold_base
        self.th_fac = float(threshold_factor)
        self.data = []
        self.energy = None
        self.threshold = None
        self.size = None
        self.nchan = None
        self.extracted_events = None
        self.ch_sep = bool(ch_separate)
        # properties handles
        self._events = None

        # energy function
        if energy_func is not None:
            self._energy_func = energy_func

        # threshold function
        if threshold_func is not None:
            self._threshold_func = threshold_func

    ## properties

    def _get_events(self):
        return self._events

    def get_events(self):
        return  self._get_events()

    def _set_events(self, value):
        self._events = value

    def set_events(self, value):
        self._set_events(value)

    events = property(get_events, set_events)

    ## node implementations

    def is_invertible(self):
        return False

    def is_trainable(self):
        return True

    def _reset(self):
        self.data = []
        self.energy = None
        self.threshold = None
        self.size = None
        self.nchan = None
        self.events = None
        self.extracted_events = None

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _train(self, x):
        self.data.append(x)

    def _stop_training(self, *args, **kwargs):
        # produce data in one piece
        self.data = sp.vstack(self.data)
        # calculate energy
        self.energy = self._energy_func(self.data)
        if self.energy.ndim == 1:
            self.energy = sp.atleast_2d(self.energy).T
        self.size, self.nchan = self.energy.shape

    def _execute(self, x, **kwargs):
        """calls self._apply_threshold() and return the events found"""

        # assert energy and threshold
        if self.energy is None:
            raise EnergyNotCalculatedError

        # channels separate?
        if self.ch_sep is False:
            self.energy = sp.atleast_2d(self.energy.max(axis=1)).T

        # threshold
        self._calc_threshold()

        # events
        self.events = threshold_detection(
            self.energy,
            self.threshold,
            min_dist=self.min_dist,
            mode=self.th_mode,
            find_max=self.find_max)

        # return
        return x

    ## public methods

    def get_epochs(self, cut=None, invert=False, merge=False):
        """returns epochs based on self.events for the current iteration

        :Parameters:
            cut : (int,int)
                Window size of an epoch in samples (befor,after) the event
                sample. If None, self._tf will be used.
            invert : bool
                Inverts the epochs, frex to yield noise epochs instead of
                spike
                epochs.
            merge : bool
                Merges overlapping epochs.
        :Returns:
            ndarray
                ndarray with epochs on the rows [[start,end]]
        """

        # checks
        if self.events is None or self.events.size == 0:
            # do we have events yet?
            return sp.zeros((0, 2), dtype=INDEX_DTYPE)
        if cut is None:
            cut = get_cut(self.tf)
        else:
            cut = get_cut(cut)

        # calc epochs
        if invert is True:
            rval = sp.vstack((
                sp.concatenate(([0], self.events + cut[1])),
                sp.concatenate((self.events - cut[0], [self.size]))
                )).T
        else:
            rval = sp.vstack((
                self.events - cut[0],
                self.events + cut[1]
                )).T

        # check for merges
        if merge is True:
            rval = merge_epochs(rval)

        # return stuff
        if rval.dtype is not INDEX_DTYPE:
            rval = rval.astype(INDEX_DTYPE)
        return rval

    def get_extracted_events(self, mc=False, align_at=-1, kind='min',
                             rsf=1.0, buffer=False):
        """yields the extracted spikes

        :type mc: bool
        :param mc: if True, return multichannel events, else return
            concatenated events.
            Default=False
        :type align_at: int or float
        :param align_at: if a float from (0.0,1.0), determine the align_sample
            according to that weight. If a positive integer from (0,
            self.tf-1] use that sample as the align_sample.
            Default=0.25 * self.tf
        :type kind: str
        :param kind: one of "min", "max", "energy" or "none". method
            to use for alignment, will be passed to the alignment function.
            Default='min'
        :type rsf: float
        :param rsf: resampling factor (use integer values of powers of 2)
        :type buffer: bool
        :param buffer: if True, write to buffer regardless of current buffer
            state.
            Default=False
        """

        if self.events is None:
            if mc is True:
                size = 0, self.tf, self.nchan
            else:
                size = 0, self.tf * self.nchan
            return sp.zeros(size)
            #raise ValueError('no events present!')

        if self.extracted_events is None or buffer:
            if align_at < 0:
                align_at = .25
            if isinstance(align_at, float):
                if 0.0 <= align_at <= 1.0:
                    align_at *= self.tf
                align_at = int(align_at)
            self.extracted_events, self.events = get_aligned_spikes(
                self.data, self.events, align_at=align_at, tf=self.tf, mc=mc,
                kind=kind, rsf=rsf)

        # return extracted events
        return self.extracted_events

    ## internal methods

    def _energy_func(self, x):
        """energy operator to apply to the input signal

        Overwrite this method in subclasses, default behaviour: identity

        This method calculates the energy to use during the feature detection.
        This can be any operator that maps the input signal [x] into a
        signal of
        equal shape. Do not set any members here, just return the result of
        the
        energy operator applied to the input signal.
        """

        return x

    def _threshold_func(self, x):
        """method of threshold calculation

        Overwrite this method in subclasses, default behaviour: zero

        This method calculates the threshold to use during feature detection
        . It
        will be applied to each channel individually and must return a scalar
        when called with x, which is a ndim=1 ndarray.
        """

        return 0.0

    def _calc_threshold(self):
        """calculates the threshold"""

        base = {
                   'signal': self.data,
                   'energy': self.energy
               }[self.th_base]
        if self.ch_sep is False:
            base = sp.atleast_2d(sp.absolute(base).max(axis=1)).T
        self.threshold = sp.asarray(
            [self._threshold_func(base[:, c])
             for c in xrange(base.shape[1])], dtype=self.dtype)
        self.threshold *= self.th_fac

    def plot(self, show=False):
        """plot detection in mcdata plot"""

        try:
            from spikeplot import plt, mcdata, COLOURS
        except ImportError:
            return None

        fig = mcdata(self.data, other=self.energy, events={0: self.events},
                     show=False)
        for i, th in enumerate(self.threshold):
            fig.axes[-1].axhline(th, c=COLOURS[i % len(COLOURS)])
        self._plot_additional(fig)
        if show is True:
            plt.show()
        return fig

    def _plot_additional(self, fig):
        pass

## spike detector implementations

class SDAbsNode(ThresholdDetectorNode):
    """spike detector

    energy: absolute of the signal
    threshold: signal.std
    """

    def __init__(self, **kwargs):
        """
        :Parameters:
            see ThresholdDetectorNode
        """

        # super
        kwargs.update(energy_func=sp.absolute,
                      threshold_base='signal',
                      threshold_func=sp.std)
        super(SDAbsNode, self).__init__(**kwargs)

    def _threshold_func(self, x):
        return self.th_fac * x.std(axis=0)


class SDSqrNode(ThresholdDetectorNode):
    """spike detector

    energy: square of the signal
    threshold: signal.var
    """

    def __init__(self, **kwargs):
        """
        :Parameters:
            see ThresholdDetectorNode
        """

        # super
        kwargs.update(energy_func=sp.square,
                      threshold_base='signal',
                      threshold_func=sp.var)
        super(SDSqrNode, self).__init__(**kwargs)


class SDMteoNode(ThresholdDetectorNode):
    """spike detector

    energy: multiresolution teager energy operator
    threshold: energy.std
    """

    def __init__(self, kvalues=[1, 3, 5, 7, 9], quantile=0.98, **kwargs):
        """
        :type kvalues: list
        :param kvalues: integers determining the kteo detectors to build the
        multiresolution teo from.
        :type quantile: float
        :param quantile: quantile of the MTeo output to use for threshold
        calculation.
        """

        # super
        kwargs.update(
            threshold_base='energy',
            threshold_factor=kwargs.get('threshold_factor', 0.96),
            min_dist=kwargs.get('min_dist', 5),
            ch_separate=True)
        super(SDMteoNode, self).__init__(**kwargs)

        # members
        self.kvalues = map(int, kvalues)
        self.quantile = quantile

    def _energy_func(self, x):
        return sp.vstack([mteo(x[:, c], kvalues=self.kvalues, condense=True)
                          for c in xrange(x.shape[1])]).T

    def _threshold_func(self, x):
        return mquantiles(x, prob=[self.quantile])[0]


class SDPeakNode(ThresholdDetectorNode):
    """spike detector

    energy: absolute of the signal
    threshold: signal.std
    """

    def __init__(self, **kwargs):
        """
        :Parameters:
            see ThresholdDetectorNode
        """

        # super
        kwargs.update(threshold_base='signal',
                      threshold_func=sp.std)
        super(SDPeakNode, self).__init__(**kwargs)

    def _threshold_func(self, x):
        return self.th_fac * x.std(axis=0)


class SDKteoNode(ThresholdDetectorNode):
    """spike detector

    energy: teager energy operator
    threshold: energy.std
    """

    def __init__(self, kvalue=1, quantile=0.98, **kwargs):
        """
        :Parameters:
            see ThresholdDetectorNode

            kvalue : int
                Integer determining the kteo detector resolution.
        """

        # super
        kwargs.update(
            threshold_base='energy',
            threshold_factor=kwargs.get('threshold_factor', 0.96),
            min_dist=kwargs.get('min_dist', 5),
            ch_separate=True)
        super(SDKteoNode, self).__init__(**kwargs)

        # members
        self.kvalue = int(kvalue)
        self.quantile = quantile

    def _energy_func(self, x):
        return sp.vstack([kteo(x[:, c], k=self.kvalue)
                          for c in xrange(x.shape[1])]).T

    def _threshold_func(self, x):
        return mquantiles(x, prob=[self.quantile])[0]


class SDIntraNode(ThresholdDetectorNode):
    """spike detector

    energy: identity
    threshold: zero
    """

    def __init__(self, **kwargs):
        """
        :Parameters:
            see ThresholdDetectorNode
        """

        # super
        kwargs.update(threshold_base='signal')
        super(SDIntraNode, self).__init__(**kwargs)

##--- MAIN

if __name__ == '__main__':
    pass
