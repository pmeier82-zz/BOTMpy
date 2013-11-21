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

"""sorting node base class"""
__docformat__ = "restructuredtext"
__all__ = ["FilterBankSortingNode"]

## IMPORTS

import copy
import logging
import scipy as sp
from ...common import overlaps, dict_list_to_ndarray, get_cut, get_aligned_spikes
from ..spike_detection import SDMneoNode
from .filter_bank import FilterBankError, FilterBankNode

## CONSTANTS

MTEO_DET = SDMneoNode
MTEO_KWARGS = {'kvalues': [3, 9, 15, 21],
               'threshold_factor': 0.98,
               'min_dist': 32}

## CLASSES

class FilterBankSortingNode(FilterBankNode):
    """abstract class that handles filter instances and their outputs

    This class provides a pipeline structure to implement spike sorting
    algorithms that operate on a filter bank. The implementation is done by
    implementing the `self._pre_filter`, `self._post_filter`, `self._pre_sort`,
    `self._sort_chunk` and `self._post_sort` methods with meaning full
    processing. After the filter steps the filter output is present and can be
    processed on. Input data can be partitioned into chunks of smaller size.
    """

    def __init__(self, **kwargs):
        """
        :type ce: TimeSeriesCovE
        :keyword ce: covariance estimator instance, if None a new instance
            will be created and initialised with the identity matrix
            corresponding to the template size.
            Required
        :type templates: ndarray
        :keyword templates: templates to initialise the filter stack.
            [ntemps][tf][nc] a tensor of templates
        :type align_kind: str
        :keyword align_kind: The feature used for alignment. One of:

        - "max"    - align on maximum of the waveform
        - "min"    - align on minimum of the waveform
        - "energy" - align on peak of energy
        - "none"   - no alignment

        Default='min'
        :type chan_set: tuple
        :keyword chan_set: tuple of int designating the subset of channels
            this filter bank operates on.
            Default=tuple(range(nc))
        :type filter_cls: LinearFilterNode
        :keyword filter_cls: the class of filter node to use for the filter
            bank, this must be a subclass of 'FilterNode'.
            Default=MatchedFilterNode
        :type rb_cap: int
        :keyword rb_cap: capacity of the ringbuffer that stores observations
            to calculate the mean template.
            Default=350
        :type chunk_size: int
        :keyword chunk_size: if input data will be longer than chunk_size, the
            input will be processed chunk per chunk to overcome memory sinks
            Default=100000
        :type verbose: int
        :keyword verbose: verbosity level, 0:none, >1: print .. ref `VERBOSE`
                Default=0
        :type dtype: dtype resolvable
        :keyword dtype: anything that resolves into a scipy dtype, like a
            string or number type
            Default=None
        """

        # kwargs
        templates = kwargs.pop('templates', None)
        tf = kwargs.get('tf', None)
        self._align_kind = kwargs.pop('align_kind', 'min')
        if tf is None and templates is None:
            raise FilterBankError('\'templates\' or \'tf\' are required!')
        if tf is None:
            if templates.ndim != 3:
                raise FilterBankError(
                    'templates have to be provided in a tensor of shape '
                    '[ntemps][tf][nc]!')
            kwargs['tf'] = templates.shape[1]
        chunk_size = kwargs.pop('chunk_size', 100000)
        # everything not popped goes to super
        super(FilterBankSortingNode, self).__init__(**kwargs)

        # members
        self._fout = None
        self._data = None
        self._chunk = None
        self._chunk_offset = 0
        self._chunk_size = int(chunk_size)
        self.rval = {}

        # create filters for templates
        if templates is not None:
            for temp in templates:
                self.filter_create(temp)
            self._update_internals()

    ## SortingNode interface

    def _execute(self, x):
        # No channel masking for now
        #self._data = x[:, self._chan_set]
        self._data = x
        dlen = self._data.shape[0]
        self.rval.clear()
        for i in self._idx_active_set:
            self.rval[i] = []
        curr_chunk = 0
        has_next_chunk = True

        # sort per chunk
        while has_next_chunk:
            # get chunk limits
            self._chunk_offset = curr_chunk * self._chunk_size
            clen = min(dlen, (curr_chunk + 1) * self._chunk_size)
            clen -= self._chunk_offset

            # generate data chunk and process
            self._chunk = self._data[
                          self._chunk_offset:self._chunk_offset + clen]
            self._fout = sp.empty((clen, self.nf))

            # filtering
            self._pre_filter()
            self._fout = super(FilterBankSortingNode, self)._execute(
                self._chunk)
            self._post_filter()

            # sorting
            self._pre_sort()
            self._sort_chunk()
            self._post_sort()

            # iteration
            curr_chunk += 1
            if self._chunk_offset + clen >= dlen:
                has_next_chunk = False
        self._combine_results()

        # return input data
        return x

    ## FilterBankSortingNode interface - prototypes

    def _pre_filter(self):
        pass

    def _post_filter(self):
        pass

    def _pre_sort(self):
        pass

    def _post_sort(self):
        pass

    def _sort_chunk(self):
        pass

    def _combine_results(self):
        self.rval = dict_list_to_ndarray(self.rval)
        correct = int(self._tf / 2)
        for k in self.rval:
            self.rval[k] -= correct

    ## result access

    def spikes_u(self, u, mc=True, exclude_overlaps=True, overlap_window=None,
                 align_at=-1, align_kind='min', align_rsf=1.):
        """yields the spike for the u-th filter

        :type u: int
        :param u: index of the filter # CHECK THIS
        :type mc: bool
        :param mc: if True, return spikes multi-channeled, else return spikes
            concatenated
            Default=True
        :type exclude_overlaps: bool
        :param exclude_overlaps: if True, exclude overlap spike
        :type overlap_window: int
        :param overlap_window: if `exclude_overlaps` is True, this will define
            the overlap range,
            if None set overlap_window=self._tf.
            Default=None
        """

        # init
        cut = get_cut(self._tf)
        rval = None
        size = 0, sum(cut), self._data.shape[1]
        if mc is False:
            size = size[0], size[1] * size[2]
        st_dict = copy.deepcopy(self.rval)

        # extract spikes
        spks, st_dict[u] = get_aligned_spikes(
            self._data,
            st_dict[u],
            align_at=align_at,
            tf=self._tf,
            mc=mc,
            kind=align_kind,
            rsf=align_rsf)
        if exclude_overlaps is True:
            if st_dict[u].size > 0:
                ovlp_info = overlaps(st_dict, overlap_window or self._tf)[0]
                spks = spks[ovlp_info[u] == False]
        return spks

    ## plotting methods

    def plot_sorting(self, ph=None, show=False):
        """plot the sorting of the last data chunk

        :type ph: plot handle
        :param ph: plot handle top use for the plot
        :type show: bool
        :param show: if True, call plt.show()
        """

        # get plotting tools
        try:
            from spikeplot import COLOURS, mcdata
        except ImportError:
            return None

        # check
        if self._data is None or self.rval is None or len(
                self._idx_active_set) == 0:
            logging.warn('not initialised properly to plot a sorting!')
            return None

        # create events
        ev = {}
        if self.rval is not None:
            temps = self.template_set
            for i in self._idx_active_set:
                if i in self.rval:
                    if self.rval[i].any():
                        ev[i] = (self.bank[i].xi, self.rval[i])

        # create colours
        cols = COLOURS[:self.nf]

        # calc discriminants for single units
        other = None
        if self.nf > 0:
            self.reset_history()
            other = super(FilterBankSortingNode, self)._execute(self._data)
            other += getattr(self, '_lpr_s', sp.log(1.e-6))
            other -= [.5 * self.get_xc_at(i)
                      for i in xrange(self.nf)]

        # plot mcdata
        return mcdata(self._data, other=other, events=ev,
                      plot_handle=ph, colours=cols, show=show)

    def plot_sorting_waveforms(self, ph=None, show=False, **kwargs):
        """plot the waveforms of the sorting of the last data chunk

        :type ph: plot handle
        :param ph: plot handle to use for the
        :type show: bool
        :param show: if True, call plt.show()
        """

        # get plotting tools
        try:
            from spikeplot import waveforms
        except ImportError:
            return None

        # check
        if self._data is None or self.rval is None or len(
                self._idx_active_set) == 0:
            logging.warn('not initialised properly to plot a sorting!')
            return None

        # init
        wf = {}
        temps = {}
        cut = get_cut(self._tf)

        # build waveforms
        for u in self.rval:
            spks_u = self.spikes_u(
                u, exclude_overlaps=False, align_kind=self._align_kind,
                align_at=getattr(self, '_learn_templates', -1),
                align_rsf=getattr(self, '_learn_templates_rsf', 1.))
            temps[u] = self.bank[u].xi_conc
            if spks_u.size > 0:
                wf[u] = self.spikes_u(u, align_kind=self._align_kind)
            else:
                wf[u] = temps[u]

        """
        waveforms(waveforms, samples_per_second=None, tf=None, plot_mean=False,
              plot_single_waveforms=True, set_y_range=False,
              plot_separate=True, plot_handle=None, colours=None, title=None,
              filename=None, show=True):
        """
        return waveforms(wf, samples_per_second=None, tf=self._tf,
                         plot_mean=True, templates=temps,
                         plot_single_waveforms=True, set_y_range=False,
                         plot_separate=True, plot_handle=ph, show=show)

    def sorting2gdf(self, fname):
        """yield the gdf representing the current sorting"""

        GdfFile.write_gdf(fname, self.rval)

## MAIN

if __name__ == "__main__":
    pass

## EOF
