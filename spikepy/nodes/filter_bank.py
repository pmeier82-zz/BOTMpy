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

"""implementation of a filter bank consisting of a set of filters"""

__docformat__ = 'restructuredtext'
__all__ = ['FilterBankError', 'FilterBankNode']

##---IMPORTS

import scipy as sp
from spikeplot import waveforms, xvf_tensor
from .base_nodes import Node
from .linear_filter import FilterNode
from ..common import (TimeSeriesCovE, xi_vs_f, overlaps, dict_list_to_ndarray,
                      epochs_from_spiketrain_set, get_cut, get_aligned_spikes)

##---CLASSES

class FilterBankError(Exception):
    pass


class FilterBankNode(Node):
    """abstract class that handles filter instances and their outputs

    All filters constituting the filter bank have to be of the same temporal
    extend (Tf) and process the same channel set.
    """

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
        :type debug: bool
        :keyword debug: if True, store intermediate results and generate
            verbose output
            Default=False
        """

        # kwargs
        ce = kwargs.pop('ce', None)
        chan_set = kwargs.pop('chan_set', None)
        filter_cls = kwargs.pop('filter_cls', None)
        rb_cap = kwargs.pop('rb_cap', 350)
        tf = kwargs.pop('tf', 47)
        debug = kwargs.pop('debug', False)
        # everything else in kwargs goes to mdp.Node.__init__ via super

        # checks
        if not issubclass(ce.__class__, TimeSeriesCovE):
            raise TypeError('\'ce\' of type TimeSeriesCovE is required!')
        if not issubclass(filter_cls.__class__, FilterNode):
            raise TypeError('\'filter_cls\' of type FilterNode is required!')
        if chan_set is None:
            chan_set = tuple(range(ce.get_nc()))

        # super
        super(FilterBankNode, self).__init__(**kwargs)

        # members
        self._chan_set = tuple(sorted(chan_set))
        self._tf = int(tf)
        self._nc = len(self._chan_set)
        self._filter_cls = filter_cls
        self._rb_cap = int(rb_cap)
        self._xcorrs = None
        self._ce = None
        self._is_initialised = False
        self.debug = bool(debug)
        self.bank = []
        self.ce = ce

    ## properties

    def get_tf(self):
        return self._tf

    tf = property(get_tf)

    def get_nc(self):
        return self._nc

    nc = property(get_nc)

    def get_chan_set(self):
        return self._chan_set

    cs = property(get_chan_set)

    def get_ce(self):
        return self._ce

    def set_ce(self, value):
        if not issubclass(value.__class__, TimeSeriesCovE):
            raise TypeError('Has to be of type %s' % TimeSeriesCovE)
        if value.tf_max < self._tf:
            raise ValueError('tf_max of cov_est is < than filter bank tf')

        # TODO: not sure how to solve this
        #if value.get_nc() < self._nc:
        #    raise ValueError('nc of cov_est is < than the filter bank nc')
        self._ce = value
        self._check_internals()

    ce = property(get_ce, set_ce)

    def get_nfilter(self):
        return len(self.bank)

    nfilter = property(get_nfilter)

    def get_template_set(self):
        if not self.bank:
            return sp.zeros((0, self._tf, self._nc), dtype=self.dtype)
        return sp.asarray([f.xi for f in self.bank])

    template_set = property(get_template_set)

    def get_template_set_conc(self):
        if not self.bank:
            return sp.zeros((0, self._tf, self._nc), dtype=self.dtype)
        return sp.asarray([f.xi_conc for f in self.bank])

    template_set_conc = property(get_template_set_conc)

    def get_filter_set(self):
        if not self.bank:
            return sp.zeros((0, self._tf, self._nc), dtype=self.dtype)
        return sp.asarray([f.f for f in self.bank])

    filter_set = property(get_filter_set)

    def get_filter_set_conc(self):
        if not self.bank:
            return sp.zeros((0, self._tf * self._nc), dtype=self.dtype)
        return sp.asarray([f.f_conc for f in self.bank])

    filter_set_conc = property(get_filter_set_conc)

    ## filter bank interface

    def reset_history(self):
        """sets the history to all zeros for all filters"""

        for f in self.bank:
            f.reset_history()

    def add_filter(self, xi, check=True):
        """adds a new filter to the filter bank"""

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
        new_f.fill_xi_buf(xi)
        self.bank.append(new_f)

        # return and check internals
        rval = True
        if check is True:
            rval = self._check_internals()
        return rval

    def _check_internals(self):
        """triggers filter recalculation and rebuild xcorr tensor"""

        # check
        if self.debug:
            print '_check_internals'
        if not self.bank:
            return

        # build filters
        for f in self.bank:
            f.calc_filter()

        # build cross-correlation tensor
        self._xcorrs = xi_vs_f(self.template_set_conc,
                               self.filter_set_conc,
                               nc=self._nc)

    ## mpd.Node interface

    def is_invertible(self):
        return False

    def is_trainable(self):
        return False

    def _execute(self, x):
        if not self.bank:
            return sp.zeros((x.shape[0], 0), dtype=self.dtype)
        self._data = sp.ascontiguousarray(x, dtype=self._dtype)
        rval = sp.empty((x.shape[0], self.nfilter))
        for i in xrange(self.nfilter):
            rval[:, i] = self.bank[i](x)
        return rval

    ## plotting methods

    def plot_xvft(self, ph=None, show=False):
        """plot the Xi vs F Tensor of the filter bank"""

        inlist = [self.template_set_conc, self.filter_set_conc, self._xcorrs]
        return xvf_tensor(inlist, nc=self._nc, plot_handle=ph, show=show)

    def plot_template_set(self, ph=None, show=False):
        """plot the template set in a waveform plot"""

        units = {}
        for i in xrange(len(self.bank)):
            units[i] = self.bank[i]._xi_buf[:]

        return waveforms(units,
                         tf=self._tf,
                         plot_separate=True,
                         plot_mean=True,
                         plot_single_waveforms=True,
                         plot_handle=ph, show=show)

##---MAIN

if __name__ == '__main__':
    pass
