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
from spikepy.common import (mcfilter_hist, mcvec_from_conc, mcvec_to_conc,
                            TimeSeriesCovE, MxRingBuffer, snr_maha)

##---CLASSES

class FilterError(Exception):
    pass


class FilterNode(Node):
    """abstract classifier for patterns in timeseries data

    This node applies a linear filter to the data and returns the filtered data.
    The derivation of the filter (f) from the pattern (xi) is specified in the
    implementing subclass via the 'filter_calculation' classmethod.
    The template will be averaged from a ringbuffer of observations. The
    covariance matrix is supplied from an external covariance estimator.
    """

    ## constructor

    def __init__(self, tf, nc, ce, chan_set=(0, 1, 2, 3), rb_cap=350,
                 dtype=sp.float32):
        """
        :Parameters:
            tf : int
                the template length
                Required
            nc : int
                the template channel count
                Required
            ce : TimeSeriesCovE
                Covariance Estimator instance or None.
                Required
            chan_set : tuple
                tuple of int designating the subset of channels this filter
                operates on.
                Default=(0,1,2,3)
            rb_cap : int
                capacity of the xi buffer
                Default=350
            dtype : scipy.dtype resolvable
                Determines the internal dtype.
                Default=scipy.float32
        """

        # checks
        if tf <= 0:
            raise ValueError('tf <= 0! has to be > 0')
        if nc <= 0:
            raise ValueError('nc <= 0! has to be > 0')
        if not isinstance(ce, TimeSeriesCovE):
            raise TypeError('cov_est has to be of type TimeSeriesCovE')
        if not ce.is_initialised:
            raise ValueError('the covariance estimator has to be initialised!')

        # super
        super(FilterNode, self).__init__(output_dim=1, dtype=dtype)

        # members
        self._xi_buf = MxRingBuffer(capacity=rb_cap, dimension=(tf, nc))
        self._ce = None
        self._f = None
        self._hist = sp.zeros((tf - 1, nc), order='C', dtype=sp.float32)
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
            raise TypeError('Has to be of type %s' % TimeSeriesCovE)
        if value.get_tf_max() < self.tf:
            raise ValueError('tf_max of cov_est < than filter tf')
        if value.get_nc() < self.nc:
            raise ValueError('nc of cov_est < than filter nc')
        self._ce = value
        if len(self._xi_buf) > 0:
            self.calc_filter()

    ce = property(get_ce, set_ce)

    def get_snr(self):
        return snr_maha(sp.atleast_2d(mcvec_to_conc(self.xi)),
                        self._ce.get_icmx(tf=self.tf, chan_set=self._chan_set)
        )[0]

    snr = property(get_snr)

    ## mdp interface

    def _execute(self, x, *args, **kwargs):
        """apply the filter to data

        The filter is applied to the data. If the data has more than one channel,
        the filter outputs are summed over the channels
        """

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

        :Parameters:
            wf : ndarray
                ndarray of shape (self.tf, self.nc)
            recalc : bool
                if True, call self.calc_filter after appending
        """

        data = self._check_wf_for_xi_buf(wf)
        self._xi_buf.append(data)
        if recalc is True:
            self.calc_filter()

    def extend_xi_buf(self, wfs, recalc=False):
        """append a list of waveforms to the xi_buffer

        :Parameters:
            wfs : ndarray
                ndarray of shape (n, self.tf, self.nc)
            recalc : bool
                if True, call self.calc_filter after extending
        """

        self._check_wf_for_xi_buf(wfs[0])
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

        data = self._check_wf_for_xi_buf(wf)
        self._xi_buf.fill(data)
        if recalc is True:
            self.calc_filter()

    def _check_wf_for_xi_buf(self, wf):
        """checks the right dim and shape for a waveform"""

        if wf.ndim == 1:
            wf = mcvec_from_conc(wf, nc=self.nc)
        if wf.shape != (self.tf, self.nc):
            raise ValueError('wrong shape, assumed %s, got %s' %
                             (str((self.tf, self.nc)),
                              str(wf.shape)))
        return wf

    ## history maintenance

    def reset_history(self):
        """sets the history to all zeros"""

        self._hist[:] = 0.0

    ## filter calculation

    def calc_filter(self):
        """initiate a calculation of the filter"""

        self._f = self.filter_calculation(self.xi, self.ce, self._chan_set)

    @classmethod
    def filter_calculation(cls, xi, ce, cs):
        """ABSTRACT METHOD FOR FILTER CALCULATION

        returns f = F(xi, ce, cs), the filter given the pattern and a covariance
        matrix (constrained by the channel set for the filter), end eventually
        some normalisation factor

        Implement this in a meaningful way in a subclass. The method should
        return the filter given the inputs. The filter is usually the same size
        as the pattern xi.
        """

        raise  NotImplementedError

    ## special methods

    def __str__(self):
        return '%s(tf=%s,nc=%s,cs=%s)' % (self.__class__.__name__,
                                          self.tf, self.nc,
                                          str(self._chan_set))


class MatchedFilterNode(FilterNode):
    """matched filters are filters in the time domain, that optimise the signal
    to noise ratio with respect to a noise covariance matrix
    """

    @classmethod
    def filter_calculation(cls, xi, ce, cs):
        """calculate the matched filter for the patter xi w.r.t. the covariance
        matrix provided by by the covariance estimator
        """

        if not ce.is_initialised():
            raise FilterError('covariance estimator has not been initialised!')
        tf, nc = xi.shape
        params = {'tf':tf, 'chan_set':cs}
        if ce.is_cond_ok(**params) is True:
            icmx = ce.get_icmx(**params)
        else:
            icmx = ce.get_icmx_loaded(**params)
        return sp.ascontiguousarray(
            mcvec_from_conc(sp.dot(mcvec_to_conc(xi), icmx), nc=nc),
            dtype=sp.float32)


class NormalisedMatchedFilterNode(FilterNode):
    """matched filters are filters in the timedomain, that optimise the signal
    to noise ratio with respect to a covariance matrix

    this is identical to MatchedFilterNode with the addition, that the filter
    output is normalised, such that the response to the exact pattern gives a
    peak of unit height.
    """

    @classmethod
    def filter_calculation(cls, xi, ce, cs):
        """calculate the matched filter for the patter xi w.r.t. the covariance
        matrix provided by cov_est
        """

        if not ce.is_initialised():
            raise FilterError('covariance estimator has not been initialised!')
        tf, nc = xi.shape
        params = {'tf':tf, 'chan_set':cs}
        if ce.is_cond_ok(**params) is True:
            icmx = ce.get_icmx(**params)
        else:
            icmx = ce.get_icmx_loaded(**params)
        f = sp.dot(mcvec_to_conc(xi), icmx)
        norm_factor = sp.dot(mcvec_to_conc(xi), f)
        return sp.ascontiguousarray(
            mcvec_from_conc(f / norm_factor, nc=nc),
            dtype=sp.float32)


##---MAIN

if __name__ == '__main__':
    # imports
    from spikepy.common import mcfilter

    # setup
    TF = 10
    NC = 2
    CS = tuple(range(NC))
    xi = sp.vstack([sp.arange(TF).astype(sp.float32)] * NC).T * 0.5
    LEN = 1000
    noise = sp.randn(LEN, NC)
    ce = TimeSeriesCovE(tf_max=TF, nc=NC)
    ce.new_chan_set(CS)
    ce.update(noise)

    # HIST filter with history item using the fastest implementation available
    mf_h = NormalisedMatchedFilterNode(TF, NC, ce, chan_set=CS)
    mf_h.fill_xi_buf(xi, recalc=True)
    print mf_h
    print mf_h.xi
    print mf_h.f
    print mf_h.tf
    print mf_h.nc
    print mf_h.ce

    # build signals
    signal = sp.zeros_like(noise)
    NPOS = 3
    POS = [int(i * LEN / (NPOS + 1)) for i in xrange(1, NPOS + 1)]
    for i in xrange(NPOS):
        signal[POS[i]:POS[i] + TF] = xi
    x = signal + noise
    late = int(TF / 2 - 1)
    pad = sp.zeros(late)
    y_h_out = mf_h(x)
    y_h = sp.concatenate([y_h_out[late:], pad])

    #  plot
    from spikeplot import plt

    plt.plot(mcfilter(x, mf_h.f), label='mcfilter (scipy.correlate)',
             color='r')
    plt.plot(y_h + .02, label='mcfilter_hist (py/c)', color='g')
    plt.plot(signal + 5)
    plt.plot(x + 15)
    plt.legend()
    plt.show()
