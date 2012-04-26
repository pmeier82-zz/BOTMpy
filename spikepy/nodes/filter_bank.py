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
from .base_nodes import Node
from .linear_filter import FilterNode
from ..common import (TimeSeriesCovE, xi_vs_f, mcvec_to_conc, overlaps,
                      epochs_from_spiketrain_set, dict_list_to_ndarray,
                      get_cut, get_aligned_spikes)

##---CLASSES

class FilterBankError(Exception):
    pass


class FilterBankNode(Node):
    """abstract class that handles filter instances and their outputs

    All filters constituting the filterbank have to be of the same temporal
    extend (Tf).
    """

    def __init__(self, **kwargs):
        """
        :type ce: TimeSeriesCovE
        :keyword ce: covariance estimator instance, if None a new instance
            will be created and initialised with the identity matrix
            corresponding to the template size.
            required
        :type chan_set: tuple
        :keyword chan_set: tuple of int designating the subset of channels
            this filter bank operates on.
            Default=tuple(range(nc))
        :type filter_cls: FilterNode
        :keyword filter_cls: the class of filter node to use for the filter
            bank, this must be a subclass of 'FilterNode'.
            required
        :type rb_cap: int
        :keyword rb_cap: capacity of the ringbuffer that stored observations
            for the filters to calculate the mean template.
            Default=350
        :type debug: bool
        :keyword debug: if True, store intermediate results and generate
            verbose output
            Default=False
        :type dtype: dtype resolvable
        :keyword dtype: anything that resolves into a scipy dtype, like a
            string or number type
            Default=None
        """

        # checks and inits
        if 'ce' not in kwargs:
            raise FilterBankError('\'ce\' is required!')
        ce = kwargs.get('ce')
        chan_set = kwargs.get('chan_set')
        filter_cls = kwargs.get('filter_cls', None)
        if not filter_cls:
            raise FilterBankError('\'filter_cls\' is required!')
        rb_cap = kwargs.get('rb_cap', 350)
        adapt_templates = kwargs.get('adapt_templates', -1)
        learn_noise = kwargs.get('learn_noise', True)
        chunk_size = kwargs.get('chunk_size', 32000)
        debug = kwargs.get('debug', False)
        dtype = kwargs.get('dtype', None)

        if 'templates' not in kwargs:
            raise FilterBankError('\'templates\' are required!')
        templates = kwargs.get('templates')

        # checks
        if templates.size == 0:
            raise FilterBankError('provide at least one template!')
        if templates.ndim != 3:
            raise FilterBankError('templates have to be provided in a tensor '
                                  'like [ntemps][tf][nc]!')
        if chan_set is None:
            chan_set = tuple(range(templates.shape[2]))
        if not issubclass(filter_cls, FilterNode):
            raise TypeError('filter_cls has to be a subclass of FilterNode!')
        if ce is None:
            ce = TimeSeriesCovE.std_white_noise_init(*templates.shape[1:])
        if not isinstance(ce, TimeSeriesCovE):
            raise TypeError('ce has to be an instance of TimeSeriesCovE '
                            'or None!')

        # super
        super(FilterBankNode, self).__init__(
            input_dim=templates.shape[2],
            output_dim=templates.shape[2],
            dtype=dtype)

        # members
        self._tf = templates.shape[1]
        self._nc = templates.shape[2]
        self._chan_set = tuple(sorted(chan_set))
        self._filter_cls = filter_cls
        self._rb_cap = int(rb_cap)
        self._chunk_size = int(chunk_size)
        self._adapt_templates = int(adapt_templates)
        self._learn_noise = bool(learn_noise)
        self._xcorrs = None
        self._fout = None
        self._ce = None
        self._data = None
        self._chunk = None
        self.debug = bool(debug)
        self.bank = []
        self.ce = ce

        # add filters for templates
        for temp in templates:
            self.add_filter(temp, check=False)
        self._check_internals()

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
        if not isinstance(value, TimeSeriesCovE):
            raise TypeError('Has to be of type %s' % TimeSeriesCovE)
        if value.get_tf_max() < self._tf:
            raise ValueError('tf_max of cov_est is < than filter bank tf')
        if value.get_nc() < self._nc:
            raise ValueError('nc of cov_est is < than the filter bank nc')
        self._ce = value
        self._check_internals()

    ce = property(get_ce, set_ce)

    def get_nfilter(self):
        return len(self.bank)

    nfilter = property(get_nfilter)

    def get_template_set(self):
        if len(self.bank) == 0:
            return None
        return sp.array([f.xi for f in self.bank])

    template_set = property(get_template_set)

    def get_template_set_conc(self):
        if len(self.bank) == 0:
            return None
        return sp.array([mcvec_to_conc(f.xi) for f in self.bank])

    template_set_conc = property(get_template_set_conc)

    def get_filter_set(self):
        if self.nfilter == 0:
            return None
        return sp.array([f.f for f in self.bank])

    filter_set = property(get_filter_set)

    def get_filter_set_conc(self):
        if self.nfilter == 0:
            return None
        return sp.array([mcvec_to_conc(f.f) for f in self.bank])

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
            raise FilterBankError('template does not match the filter banks '
                                  'filter shape of %s' %
                                  str((self._tf, self._nc)))

        # build filter and add to filterbank
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
        """internal bookkeeping that assures the filter bank is up to date"""

        # check
        if self.debug:
            print '_check_internals'
        if len(self.bank) == 0:
            return

        # build filters from templates and the cross-correlation tensor
        for filt in self.bank:
            filt.calc_filter()
        self._xcorrs = xi_vs_f(self.template_set_conc,
                               self.filter_set_conc,
                               nc=self._nc)

    def _adaption(self):
        """adapt using non overlapping spikes"""

        # checks and inits
        if self._data is None or self.rval is None:
            return
        ovlp_info = overlaps(self.rval, self._tf)[0]
        cut = get_cut(self._tf)

        # adapt filters with found waveforms
        for u in self.rval:
            st = self.rval[u][ovlp_info[u] == False]
            if len(st) == 0:
                continue
            spks_u = get_aligned_spikes(self._data,
                                        st,
                                        cut,
                                        self._adapt_templates,
                                        mc=True,
                                        kind='min')[0]
            if spks_u.size == 0:
                continue
            self.bank[u].extend_xi_buf(spks_u)

        # adapt noise covariance matrix
        if self._learn_noise:
            nep = epochs_from_spiketrain_set(self.rval,
                                             cut=cut,
                                             end=self._data.shape[0])['noise']
            self._ce.update(self._data, epochs=nep)

    ## mpd.Node interface

    def is_invertible(self):
        return False

    def is_resetable(self):
        return False

    def is_trainable(self):
        return False

    ## SortingNode interface

    def _sorting(self, x, *args, **kwargs):
        # inits
        self._data = sp.ascontiguousarray(x, dtype=self._dtype)
        dlen = self._data.shape[0]
        self.rval.clear()
        for i in xrange(self.nfilter):
            self.rval[i] = []
        curr_chunk = 0
        has_next_chunk = True

        # sort per chunk
        while has_next_chunk:
            # get chunk limits
            c_start = curr_chunk * self._chunk_size
            c_stopp = min(dlen, (curr_chunk + 1) * self._chunk_size)

            # generate chunked data and sorting for this chunk
            self._chunk = self._data[c_start:c_stopp]
            self._fout = sp.empty((self._chunk.shape[0], self.nfilter))
            self._pre_filter()
            for i in xrange(len(self.bank)):
                self._fout[:, i] = self.bank[i](self._chunk)
            self._post_filter()
            self._sort_chunk(c_start)

            # iteration
            curr_chunk += 1
            if c_stopp >= dlen:
                has_next_chunk = False
        self._combine_results()

        # adaption ?
        if self._adapt_templates >= 0:
            self._adaption()

        # return input data
        return x

    ## FilterBankSorting interface

    def _pre_filter(self):
        return

    def _post_filter(self):
        return

    def _combine_results(self):
        self.rval = dict_list_to_ndarray(self.rval)
        correct = int(self._tf / 2)
        for k in self.rval:
            self.rval[k] -= correct

    def _sort_chunk(self, offset):
        return

    ## plotting methods

    def plot_xvft(self, ph=None, show=False):
        """plot the Xi vs F Tensor of the filter bank"""

        from spikeplot import xvf_tensor

        xvf_tensor([self.template_set_conc, self.filter_set_conc,
                    self._xcorrs], nc=self._nc, plot_handle=ph, show=show)

    def plot_template_set(self, ph=None, show=False):
        """plot the template set in a waveform plot"""

        from spikeplot import waveforms

        units = {}
        for i in xrange(len(self.bank)):
            units[i] = self.bank[i]._xi_buf[:]

        waveforms(units,
                  tf=self._tf,
                  plot_separate=True,
                  plot_mean=True,
                  plot_single_waveforms=True,
                  plot_handle=ph, show=show)

##---MAIN

if __name__ == '__main__':
    pass
