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

"""filter classes for linear filters in the time domain"""
__docformat__ = 'restructuredtext'
__all__ = ['FilterError', 'FilterNode', 'MatchedFilterNode',
           'NormalisedMatchedFilterNode']

##---IMPORTS

import scipy as sp
from mdp import Node
from ..common import (mcfilter_hist, mcvec_from_conc, mcvec_to_conc,
                      TimeSeriesCovE, MxRingBuffer, snr_maha)

##---CLASSES

class FilterError(Exception):
    pass


class FilterNode(Node):
    """abstract classifier for patterns in timeseries data

    This node applies a linear filter to the data and returns the filtered
    data.
    The derivation of the filter (f) from the pattern (xi) is specified in the
    implementing subclass via the 'filter_calculation' classmethod.
    The template will be averaged from a ringbuffer of observations. The
    covariance matrix is supplied from an external covariance estimator.
    """

    ## constructor

    def __init__(self, tf, nc, ce, chan_set=None, rb_cap=350,
                 dtype=None):
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
            Default=(tuple(range(nc))
        :type rb_cap: int
        :param rb_cap: capacity of the xi buffer
            Default=350
        :type dtype : dtype resolvable
        :param dtype: determines the internal dtype
            Default=float32
        """

        # checks
        if tf <= 0:
            raise ValueError('tf <= 0')
        if nc <= 0:
            raise ValueError('nc <= 0')
        if chan_set is None:
            chan_set = tuple(range(nc))

        # super
        super(FilterNode, self).__init__(output_dim=1,
                                         dtype=dtype or sp.float32)

        # members
        self._xi_buf = MxRingBuffer(capacity=rb_cap,
                                    dimension=(tf, nc),
                                    dtype=self.dtype)
        self._ce = None
        self._f = None
        self._hist = sp.empty((tf - 1, nc), dtype=self.dtype)
        self._chan_set = tuple(sorted(chan_set))

        # set covariance estimator
        self.ce = ce

        # status
        self.active = True

    ## properties

    def get_xi(self):
        return self._xi_buf.mean()

    xi = property(get_xi)

    def get_tf(self):
        return self._xi_buf.dimension[0]

    tf = property(get_tf)

    def get_nc(self):
        return self._xi_buf.dimension[1]

    nc = property(get_nc)

    def get_f(self):
        return self._f

    f = property(get_f)

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

    ce = property(get_ce, set_ce)

    def get_snr(self):
        return snr_maha(
            sp.array([mcvec_to_conc(self.xi)]),
            self._ce.get_icmx(tf=self.tf, chan_set=self._chan_set))[0]

    snr = property(get_snr)

    ## mdp.Node interface

    def _execute(self, x, *args, **kwargs):
        """apply the filter to data"""

        rval, self._hist = mcfilter_hist(x, self._f, self._hist)
        return rval

    def is_invertible(self):
        return False

    def is_trainable(self):
        return False

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    ## xi buffer updating

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

    ## history maintenance

    def reset_history(self):
        """sets the history to all zeros"""

        self._hist[:] = 0.0

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

##---MAIN

if __name__ == '__main__':
    pass
