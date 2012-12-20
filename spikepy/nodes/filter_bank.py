# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012 Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Neural Information Processing Group (NI)
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

"""implementation of a filter bank consisting of a set of filters"""
from Queue import Queue
import ctypes as ctypes
import platform
from threading import Thread

__docformat__ = 'restructuredtext'
__all__ = ['FilterBankError', 'FilterBankNode']

##---IMPORTS

import scipy as sp
import warnings
from .base_nodes import Node
from .linear_filter import FilterNode, REMF
from ..common import (TimeSeriesCovE, xi_vs_f, VERBOSE)

##---CLASSES

class FilterBankError(Exception):
    pass


class FilterBankNode(Node):
    """abstract class that handles filter instances and their outputs

    All filters constituting the filter bank have to be of the same temporal extend (Tf) and process
    the same channel set.

    There are two different index sets. One is abbreviated "idx" and one "key". The "idx" the index
    of filter in `self.bank` and thus a unique, hashable identifier. Where as the "key" an index in a
    subset of idx. Ex.: the index for list(self._idx_active_set) would be a "key".
    """

    ## constructor

    def __init__(self, **kwargs):
        """see `mdp.Node`

        :type ce: TimeSeriesCovE
        :keyword ce: covariance estimator instance, if None a new instance
            will be created and initialised with the identity matrix
            corresponding to the template size.
            required
        :type chan_set: tuple
        :keyword chan_set: tuple of int designating the subset of channels
            this filter bank operates on. Defaults to all the channels of
            the input data, as determined by the max chan_set of the covariance
            estimator.
            Default=tuple(range(nc))
        :type filter_cls: FilterNode
        :keyword filter_cls: the class of filter node to use for the filter
            bank, this must be a subclass of 'FilterNode'.
            required
        :type rb_cap: int
        :keyword rb_cap: capacity of the ringbuffer that stored observations
            for the filters to calculate the mean template.
            Default=350
        :type tf: int
        :keyword tf: temporal extend of the filters in the filter bank in
            samples.
            Default=47
        :type verbose: int
        :keyword verbose: verbosity level, 0:none, >1: print .. ref `VERBOSE`
            Default=0
        """

        # kwargs
        ce = kwargs.pop('ce', None)
        chan_set = kwargs.pop('chan_set', None)
        filter_cls = kwargs.pop('filter_cls', REMF)
        rb_cap = kwargs.pop('rb_cap', 350)
        tf = kwargs.pop('tf', 47)
        verbose = kwargs.pop('verbose', 0)
        # everything not popped goes to mdp.Node.__init__ via super

        # checks
        if not issubclass(ce.__class__, TimeSeriesCovE):
            raise TypeError('\'ce\' of type TimeSeriesCovE is required!')
        if not issubclass(filter_cls, FilterNode):
            raise TypeError('\'filter_cls\' of type FilterNode is required!')
        if chan_set is None:
            chan_set = tuple(range(ce.get_nc()))

        # super
        super(FilterBankNode, self).__init__(**kwargs)

        # members
        self._tf = int(tf)
        self._nc = None
        self._chan_set = None
        self._xcorrs = None
        self._ce = None
        self._filter_cls = filter_cls
        self._rb_cap = int(rb_cap)
        self._idx_active_set = set()
        self.bank = {}
        self.verbose = VERBOSE(verbose)

        # set members
        self.cs = chan_set
        self.ce = ce

    ## properties static or protected

    def get_tf(self):
        return self._tf

    tf = property(get_tf, doc='temporal filter extend [samples]')

    def get_nc(self):
        return self._nc

    nc = property(get_nc, doc='number of channels')

    def get_nf(self, active=True):
        if active:
            return len(self._idx_active_set)
        else:
            return len(self.bank)

    nf = property(get_nf, doc='number of filters')

    def get_template_set(self, active=True, mc=True):
        key_set = self._idx_active_set if active else set(self.bank.keys())
        if not key_set:
            shape = (0, self._tf, self._nc) if mc else (0, self._tf * self._nc)
            return sp.zeros(shape, dtype=self.dtype)
        f_list = self._get_idx_set(key_set)
        return sp.asarray([f.xi if mc else f.xi_conc for f in f_list])

    template_set = property(get_template_set, doc='template set of active filters')

    def get_filter_set(self, active=True, mc=True):
        key_set = self._idx_active_set if active else set(self.bank.keys())
        if not key_set:
            shape = (0, self._tf, self._nc) if mc else (0, self._tf * self._nc)
            return sp.zeros(shape, dtype=self.dtype)
        f_list = self._get_idx_set(key_set)
        return sp.asarray([f.f if mc else f.f_conc for f in f_list])

    filter_set = property(get_filter_set, doc='filter set of active filters')

    def get_xcorrs(self):
        return self._xcorrs

    xcorrs = property(get_xcorrs, doc='cross correlation tensor for active filters')

    def get_xcorrs_at(self, idx0, idx1=None, shift=0):
        if self._xcorrs is None:
            return None
        return self._xcorrs[idx0, idx1 or idx0, self._tf - 1 + shift]

    def get_idx_for(self, key):
        return list(self._idx_active_set)[key]

    def _get_idx_set(self, key_set):
        return [self.bank[k] for k in key_set]

    ## properties public

    def get_chan_set(self):
        return self._chan_set

    def set_chan_set(self, value):
        self._chan_set = tuple(sorted(value))
        self._nc = len(self._chan_set)

    cs = property(get_chan_set, set_chan_set)

    def get_ce(self):
        return self._ce

    def set_ce(self, value):
        if not issubclass(value.__class__, TimeSeriesCovE):
            raise TypeError('Has to be of type %s' % TimeSeriesCovE)
        if value.tf_max < self._tf:
            raise ValueError('tf_max of cov_est is < than filter bank tf')
        if self._chan_set not in value.get_chan_set():
            raise FilterBankError('\'chan_set\' not present at \'ce\'!')

        # TODO: not sure how to solve this
        #if value.get_nc() < self._nc:
        #    raise ValueError('nc of cov_est is < than the filter bank nc')
        self._ce = value
        self._check_internals()

    ce = property(get_ce, set_ce)

    ## filter bank interface

    def reset_history(self):
        """sets the history to all zeros for all filters"""

        for filt in self.bank.values():
            filt.reset_history()

    def reset_rates(self):
        """resets the rate estimators for all filters (if applicable)"""

        for filt in self.bank.values():
            if hasattr(filt, 'rate'):
                filt.rate.reset()

    def create_filter(self, xi, check=True):
        """adds a new filter to the filter bank

        :type xi: ndarray
        :param xi: template to build the filter for
        """

        # check input
        xi = sp.asarray(xi, dtype=self.dtype)
        if xi.ndim != 2 or xi.shape != (self._tf, self._nc):
            raise FilterBankError(
                'template does not match the filter banks filter shape of %s' %
                str((self._tf, self._nc)))

        # build filter and add to filter bank
        new_f = self._filter_cls(self._tf,
            self._nc,
            self._ce,
            rb_cap=self._rb_cap,
            chan_set=self._chan_set,
            dtype=self.dtype)
        #new_f.fill_xi_buf(xi)
        new_f.append_xi_buf(xi)
        idx = 0
        if len(self.bank):
            idx = max(self.bank.keys()) + 1
        self.bank[idx] = new_f
        self._idx_active_set.add(idx)

        # return and check internals
        rval = True
        if check is True:
            rval = self._check_internals()
        return rval

    def deactivate(self, idx, check=False):
        """deactivates a filter in the filter bank

        Filters are never deleted, but can be de-/reactivated and will be used
        respecting there activation state for the filter output of the
        filter bank.

        No effect if idx not in self.bank.
        """

        if idx in self.bank:
            self.bank[idx].active = False
            self._idx_active_set.discard(idx)
            if check is True:
                self._check_internals()
        else:
            warnings.warn('no idx=%s in filter bank!' % idx)

    def activate(self, idx, check=False):
        """activates a filter in the filter bank

        Filters are never deleted, but can be de-/reactivated and will be used
        respecting there activation state for the filter output of the
        filter bank.

        No effect if idx not in self.bank.
        """

        if idx in self.bank:
            self.bank[idx].active = True
            self._idx_active_set.add(idx)
            if check is True:
                self._check_internals()
        else:
            warnings.warn('no idx=%s in filter bank!' % idx)

    def _check_internals(self):
        """triggers filter recalculation and rebuild xcorr tensor"""

        # check
        if self.verbose.has_print:
            print '_check_internals'
        if not self.bank:
            return

        # build filters
        for i in self._idx_active_set:
            self.bank[i].calc_filter()

        # build cross-correlation tensor
        self._xcorrs = xi_vs_f(
            self.get_template_set(mc=False),
            self.get_filter_set(mc=False),
            nc=self._nc)

    ## mpd.Node interface

    def is_invertible(self):
        return False

    def is_trainable(self):
        return False

    def _execute(self, x):
        if not self._idx_active_set:
            return sp.zeros((x.shape[0], 0), dtype=self.dtype)
        rval = sp.empty((x.shape[0], self.nf))
        for k, i in enumerate(self._idx_active_set):
            rval[:, k] = self.bank[i](x)
        return rval

    ## plotting methods

    def plot_xvft(self, ph=None, show=False):
        """plot the Xi vs F Tensor of the filter bank"""

        # get plotting tools
        try:
            from spikeplot import xvf_tensor, plt
        except ImportError:
            return None

        # check
        if self.nf == 0:
            warnings.warn('skipping plot, no active units!')
            return None

        # init
        inlist = [self.get_template_set(mc=False),
                  self.get_filter_set(mc=False),
                  self._xcorrs]
        return xvf_tensor(inlist, nc=self._nc, plot_handle=ph, show=show)

    def plot_template_set(self, ph=None, show=False):
        """plot the template set in a waveform plot"""

        # get plotting tools
        try:
            from spikeplot import waveforms, plt
        except ImportError:
            return None

        # checks
        if self.nf == 0:
            warnings.warn('skipping plot, no active units!')
            return None

        # init
        units = {}
        for k in self._idx_active_set:
            units[k] = self.bank[k]._xi_buf[:]

        return waveforms(
            units,
            tf=self._tf,
            plot_separate=True,
            plot_mean=True,
            plot_single_waveforms=True,
            plot_handle=ph, show=show)

    def plot_template_set2(self, show=False):
        """plot the template set in a waveform plot"""

        # get plotting tools
        try:
            from spikeplot import plt
        except ImportError:
            return None

        # checks
        if self.nf == 0:
            warnings.warn('skipping plot, no active units!')
            return None

        # init
        f = plt.figure()
        y_min, y_max = 0, 0
        share = None
        for k, i in enumerate(self._idx_active_set):
            ax = f.add_subplot(self.nf, 1, k + 1, sharex=share, sharey=share)
            a, b = self.bank[i].plot_buffer_to_axis(axis=ax, idx=i)
            y_min = min(y_min, a)
            y_max = max(y_max, b)
            share = ax
        f.axes[0].set_ylim(y_min, y_max)

        if show is True:
            plt.show()
        return f

    def plot_filter_set(self, ph=None, show=False):
        """plot the filter set in a waveform plot"""

        # get plotting tools
        try:
            from spikeplot import waveforms
        except ImportError:
            return None

        # checks
        if self.nf == 0:
            warnings.warn('skipping plot, no active units!')
            return None

        # init
        units = {}
        for k in self._idx_active_set:
            units[k] = sp.atleast_2d(self.bank[k].f_conc)

        return waveforms(
            units,
            tf=self._tf,
            plot_separate=True,
            plot_mean=False,
            plot_single_waveforms=False,
            plot_handle=ph, show=show)

    ## special methods

    __len__ = get_nf

##---MAIN

if __name__ == '__main__':
    pass
