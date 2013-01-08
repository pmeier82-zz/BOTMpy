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

"""filter classes for linear filters in the time domain"""
__docformat__ = 'restructuredtext'
__all__ = ['FilterError', 'FilterNode', 'MatchedFilterNode',
           'NormalisedMatchedFilterNode']

##---IMPORTS

import scipy as sp
from .base_nodes import Node
from ..common import (mcfilter_hist, mcvec_from_conc, mcvec_to_conc,
                      TimeSeriesCovE, MxRingBuffer, snr_maha)

##---CLASSES

class FilterError(Exception):
    pass


class FilterNode(Node):
    """linear filter in the time domain

    This node applies a linear filter to the data and returns the filtered
    data. The derivation of the filter (f) from the pattern (xi) is
    specified in the implementing subclass via the 'filter_calculation'
    classmethod. The template will be averaged from a ringbuffer of
    observations. The covariance matrix is supplied from an external
    covariance estimator.
    """

    ## constructor

    def __init__(self, tf, nc, ce, chan_set=None, rb_cap=None, dtype=None):
        """
        :type tf: int
        :param tf: template length in samples
        :type nc: int
        :type nc: template channel count
        :type ce: TimeSeriesCovE
        :param ce: covariance estimator instance
        :type chan_set: tuple
        :param chan_set: tuple of int designating the subset of channels this
            filter operates on.
            Default=tuple(range(nc))
        :type rb_cap: int
        :param rb_cap: capacity of the xi buffer
            Default=350
        :type dtype: dtype resolvable
        :param dtype: determines the internal dtype
            Default=None
        """

        # checks
        if tf <= 0:
            raise ValueError('tf <= 0')
        if nc <= 0:
            raise ValueError('nc <= 0')
        if chan_set is None:
            chan_set = tuple(range(nc))

        # super
        super(FilterNode, self).__init__(output_dim=1, dtype=dtype)

        # members
        self._xi_buf = MxRingBuffer(capacity=rb_cap or 350, dimension=(tf, nc), dtype=self.dtype)
        self._ce = None
        self._f = None
        self._hist = sp.zeros((tf - 1, nc), dtype=self.dtype)
        self._chan_set = tuple(sorted(chan_set))
        self.ce = ce
        self.active = True

    ## properties static or protected

    def get_xi(self):
        return self._xi_buf.mean()

    xi = property(get_xi, doc='template (multi-channeled)')

    def get_xi_conc(self):
        return mcvec_to_conc(self._xi_buf.mean())

    xi_conc = property(get_xi_conc, doc='template (concatenated)')

    def get_tf(self):
        return self._xi_buf.dimension[0]

    tf = property(get_tf, doc='temporal extend [sample]')

    def get_nc(self):
        return self._xi_buf.dimension[1]

    nc = property(get_nc, doc='number of channels')

    def get_f(self):
        return self._f

    f = property(get_f, doc='filter (multi-channeled)')

    def get_f_conc(self):
        return mcvec_to_conc(self._f)

    f_conc = property(get_f_conc, doc='filter (concatenated)')

    ## properties public

    def get_ce(self):
        return self._ce

    def set_ce(self, value):
        if not isinstance(value, TimeSeriesCovE):
            raise TypeError('ce is not of type TimeSeriesCovE')
        if value.get_tf_max() < self.tf:
            raise ValueError('tf_max of ce < than filter tf')
        if value.get_nc() < self.nc:
            raise ValueError('nc of cov_est < than filter nc')
        if value.is_initialised is False:
            raise ValueError('ce not initialised!')
        self._ce = value
        if len(self._xi_buf) > 0:
            self.calc_filter()

    ce = property(get_ce, set_ce, doc='covariance estimator')

    def get_snr(self):
        return snr_maha(
            sp.array([mcvec_to_conc(self.xi)]),
            self._ce.get_icmx(tf=self.tf, chan_set=self._chan_set))[0]

    snr = property(get_snr, doc='signal to noise ratio (mahalanobis distance)')

    ## mdp.Node interface

    def _execute(self, x):
        """apply the filter to data"""

        # DOC: ascontiguousarray is here for ctypes/cython purposes
        x_in = sp.ascontiguousarray(x, dtype=self.dtype)[:, self._chan_set]
        rval, self._hist = mcfilter_hist(x_in, self._f, self._hist)
        return rval

    def is_invertible(self):
        return False

    def is_trainable(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    ## filter interface

    def append_xi_buf(self, wf, recalc=False):
        """append one waveform to the xi_buffer

        :type wf: ndarray
        :param wf: wavefom data [self.tf, self.nc]
        :type recalc: bool
        :param recalc: if True, call self.calc_filter after appending
        """

        self._xi_buf.append(wf)
        if recalc is True:
            self.calc_filter()

    def extend_xi_buf(self, wfs, recalc=False):
        """append an iterable of waveforms to the xi_buffer

        :type wfs: iterable of ndarray
        :param wfs: wavefom data [n][self.tf, self.nc]
        :type recalc: bool
        :param recalc: if True, call self.calc_filter after extending
        """

        self._xi_buf.extend(wfs)
        if recalc is True:
            self.calc_filter()

    def fill_xi_buf(self, wf, recalc=False):
        """fill all of the xi_buffer with wf

        :Parameters:
            wf : ndarrsay
                ndarray of shape (self.tf, self.nc)
            recalc : bool
                if True, call self.calc_filter after appending
        """

        self._xi_buf.fill(wf)
        if recalc is True:
            self.calc_filter()

    def reset_history(self):
        """sets the history to all zeros"""

        self._hist[:] = 0.0

    ## plotting methods

    def plot_buffer_to_axis(self, axis=None, idx=None, limits=None):
        """plots the current buffer on the passed axis handle"""

        try:
            from spikeplot import plt, COLOURS
        except ImportError:
            return None

        # init
        ax = axis
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(111)
        col = 'k'
        if idx is not None:
            col = COLOURS[idx % len(COLOURS)]
        spks = self._xi_buf[:]
        n, s, c = spks.shape
        spks = spks.swapaxes(2, 1).reshape(n, s * c)

        # plot
        ax.plot(spks.T, color='gray')
        ax.plot(spks.mean(axis=0), color=col, lw=2)
        for i in xrange(1, c):
            ax.axvline((self.tf * i), ls='dashed', color='y')
        ax.set_xlim(0, s * c)
        if limits is not None:
            ax.set_ylim(*limits)
        ax.set_xlabel('time [samples]')
        ax.set_ylabel('amplitude [mV]')

        return spks.min(), spks.max()

    ## filter calculation

    def calc_filter(self):
        """initiate a calculation of the filter"""

        self._f = self.filter_calculation(self.xi, self._ce, self._chan_set)

    @classmethod
    def filter_calculation(cls, xi, ce, cs, *args, **kwargs):
        """ABSTRACT METHOD FOR FILTER CALCULATION

        Implement this in a meaningful way in any subclass. The method should
        return the filter given the multichanneled template `xi`, the
        covariance estimator `ce` and the channel set `cs` plus any number
        of optional arguments and keywords. The filter is usually the same
        shape as the pattern `xi`.
        """

        raise  NotImplementedError

    ## special methods

    def __str__(self):
        return '%s(tf=%s,nc=%s,cs=%s)' % (self.__class__.__name__,
                                          self.tf, self.nc,
                                          str(self._chan_set))


class MatchedFilterNode(FilterNode):
    """matched filters in the time domain optimise the signal to noise ratio
    (SNR) of the matched pattern with respect to covariance matrix
    describing the noise background (deconvolution).
    """

    @classmethod
    def filter_calculation(cls, xi, ce, cs, *args, **kwargs):
        tf, nc = xi.shape
        ## don't do loading for now
        # params = {'tf':tf, 'chan_set':cs}
        # if ce.is_cond_ok(**params) is True:
        #     icmx = ce.get_icmx(**params)
        # else:
        #     icmx = ce.get_icmx_loaded(**params)
        ##
        icmx = ce.get_icmx(tf=tf, chan_set=cs)
        f = sp.dot(mcvec_to_conc(xi), icmx)
        return sp.ascontiguousarray(mcvec_from_conc(f, nc=nc),
            dtype=xi.dtype)


class NormalisedMatchedFilterNode(FilterNode):
    """matched filters in the time domain optimise the signal to noise ratio
    (SNR) of the matched pattern with respect to covariance matrix
    describing the noise background (deconvolution). Here the deconvolution
    output is normalised s.t. the response of the pattern is peak of unit
    amplitude.
    """

    @classmethod
    def filter_calculation(cls, xi, ce, cs, *args, **kwargs):
        tf, nc = xi.shape
        ## don't do loading for now
        # params = {'tf':tf, 'chan_set':cs}
        # if ce.is_cond_ok(**params) is True:
        #     icmx = ce.get_icmx(**params)
        # else:
        #     icmx = ce.get_icmx_loaded(**params)
        ##
        icmx = ce.get_icmx(tf=tf, chan_set=cs)
        f = sp.dot(mcvec_to_conc(xi), icmx)
        norm_factor = sp.dot(mcvec_to_conc(xi), f)
        return sp.ascontiguousarray(mcvec_from_conc(f / norm_factor, nc=nc),
            dtype=sp.float32)


class RateEstimator(object):
    def __init__(self, *args, **kwargs):
        self._spike_count = 0
        self._sample_count = 0
        self._n_updates_since = 0
        self._sample_rate = float(kwargs.pop('sample_rate', 32000.0))

    def estimate(self):
        return self._sample_rate * self._spike_count / self._sample_count

    def observation(self, nobs, tlen):
        self._spike_count += nobs
        self._sample_count += tlen
        if nobs > 0:
            self._n_updates_since = 0
        else:
            self._n_updates_since += 1

    def reset(self):
        self._spike_count = 0
        self._sample_count = 0
        self._n_updates_since = 0


class REMF(MatchedFilterNode):
    def __init__(self, *args, **kwargs):
        srate = kwargs.pop('sample_rate', 32000.0)
        super(REMF, self).__init__(*args, **kwargs)
        self.rate = RateEstimator(srate)


class RENMF(NormalisedMatchedFilterNode):
    def __init__(self, *args, **kwargs):
        srate = kwargs.pop('sample_rate', 32000.0)
        super(RENMF, self).__init__(*args, **kwargs)
        self.rate = RateEstimator(srate)

##---MAIN

if __name__ == '__main__':
    pass
