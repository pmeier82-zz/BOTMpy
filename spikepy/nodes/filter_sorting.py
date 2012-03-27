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

"""implementation of spikesorting with optimal linear filters

See:
[1] F. Franke, M. Natora, C. Boucsein, M. Munk, and K. Obermayer. An online
spike detection and spike classification algorithm capable of instantaneous
resolution of overlapping spikes. Journal of Computational Neuroscience, 2009
[2] F. Franke, ... , Klaus Obermayer, 2012,
The revolutionary BOTM Paper
"""

__docformat__ = 'restructuredtext'
__all__ = ['FilterBankError', 'FilterBankSortingNode', 'BOTMNode']

##---IMPORTS

import scipy as sp
from scipy import linalg as sp_la
from .sorting_nodes import SortingNode
from .filter_nodes import FilterNode, MatchedFilterNode
from ..common import (TimeSeriesCovE, xi_vs_f, mcvec_to_conc, overlaps,
                      epochs_from_spiketrain_set, shifted_matrix_sub,
                      epochs_from_binvec, merge_epochs, matrix_argmax,
                      dict_list_to_ndarray, get_cut, get_aligned_spikes)

# for paralell mixin
from Queue import Queue
from threading import Thread
import ctypes as ctypes
import platform
import warnings

##---CLASSES

class FilterBankError(Exception):
    pass


class FilterBankSortingNode(SortingNode):
    """abstract class that handles filter instances and their outputs"""

    def __init__(self, **kwargs):
        """
        :type templates: ndarray
        :keyword templates: templates to initialise the filter stack.
            [ntemps][tf][nc] a tensor of templates
            Required
        :type ce: TimeSeroesCovE
        :keyword ce: covariance estimator instance, if None a new instance
            will be created and initialised with the identity matrix
            corresponding to the template size.
            Required
        :type chan_set: tuple
        :keyword chan_set: tuple of int designating the subset of channels
            this filter bank operates on.
            Default=tuple(range(nc))
        :type filter_cls: FilterNode
        :keyword filter_cls: the class of filter node to use for the filter
            bank, this must be a subclass of 'FilterNode'.
            Default=MatchedFilterNode
        :type rb_cap: int
        :keyword rb_cap: capacity of the ringbuffer that stored observations
            for the filters to calculate the mean template.
            Default=350
        :type adapt_templates: int
        :keyword adapt_templates: if non-negative integer, adapt the filters
            with the found events aligned at that sample.
            Default=-1
        :type learn_noise: bool
        :keyword learn_noise: if True, adapt the noise covariance matrix with
            the noise epochs w.r.t. the found events. Else, do not learn the
            noise.
            Default=True
        :type chunk_size: int
        :keyword chunk_size: if input data will be longer than chunk_size, the
            input will be processed chunk per chunk to overcome memory sinks
            Default=32000
        :type debug: bool
        :keyword debug: if True, store intermediate results and generate
            verbose output
            Default=False
        :type dtype: dtype resolvable
        :keyword dtype: anything that resolves into a scipy dtype, like a
            string or number type
            Default=None
        """

        # process kwargs
        if 'templates' not in kwargs:
            raise FilterBankError('\'templates\' are required!')
        templates = kwargs.get('templates')
        if 'ce' not in kwargs:
            raise FilterBankError('\'ce\' is required!')
        ce = kwargs.get('ce')
        chan_set = kwargs.get('chan_set')
        filter_cls = kwargs.get('filter_cls', MatchedFilterNode)
        rb_cap = kwargs.get('rb_cap', 350)
        adapt_templates = kwargs.get('adapt_templates', -1)
        learn_noise = kwargs.get('learn_noise', True)
        chunk_size = kwargs.get('chunk_size', 32000)
        debug = kwargs.get('debug', False)
        dtype = kwargs.get('dtype', None)

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
        super(FilterBankSortingNode, self).__init__(
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


class ParallelAdaptionMixIn(object):
    """mixin class to send the check internals into a separate thread

    Provides old behavior under same name and adds the methods for paralell
    execution.
    """

    def _check_internals(self):
        """internal bookkeeping that assures the filter bank is up to date"""

        # check
        if self.debug:
            print '_check_internals (paralell)'
        if len(self.bank) == 0:
            return

        # parallel execution
        self._check_internals_par()
        self._check_internals_t.join()
        good = False
        while not good:
            good = self._check_internals_par_collect()

    def _check_internals_par(self):
        """check internals trigger function"""

        if self.debug:
            print 'Starting check_internals_par!'
        self._check_internals_q = Queue()
        self._check_internals_t = Thread(
            target=self._check_internals_par_kernel,
            args=(self._check_internals_q,))
        self._check_internals_t.start()

    def _check_internals_par_kernel(self, q):
        """check_internals kernel for the background thread"""

        # for windows: reduce the priority of the current thread to idle
        if platform.system() == 'Windows':
            THREAD_PRIORITY_IDLE = -15
            THREAD_SET_INFORMATION = 0x20
            w32 = ctypes.windll.kernel32
            handle = w32.OpenThread(THREAD_SET_INFORMATION, False,
                                    w32.GetCurrentThreadId())
            result = w32.SetThreadPriority(handle, THREAD_PRIORITY_IDLE)
            w32.CloseHandle(handle)
            if not result:
                print 'Failed to set priority of thread', w32.GetLastError()
        self._check_internals()
        q.put((self.bank, self._xcorrs))

    def _check_internals_par_collect(self):
        """returns true if collect was good"""

        if self._check_internals_q.empty():
            return False
        else:
            self.bank, self._xcorrs = self._check_internals_q.get()
            self._check_internals_q.task_done()
            self._check_internals_q = None
            return True


class BOTMNode(FilterBankSortingNode):
    """FilterBanksSortingNode derivative for the BOTM algorithm

    Can use two implementations of the Bayes Optimal Template-Matching (BOTM)
    algorithm as presented in [2]. First implementation uses explicitly
    constructed overlap channels for the extend of the complete input
    signal, the other implementation uses subtractive interference
    cancellation (SIC) on epochs of the signal, where the template
    discriminants are greater the the noise discriminant.
    """

    ## constructor

    def __init__(self, **kwargs):
        """
        :type ovlp_taus: list
        :keyword ovlp_taus: None or list of tau values. If list of tau
            values is given, discriminant-functions for all pair-wise
            template overlap cases with the given tau values will be created
            and evaluated. If None a greedy subtractive interference
            cancellation (SIC) approach will be used.
            Default=None
        :type spk_pr: float
        :keyword spk_pr: spike prior value
            Default=1e-6
        :type noi_pr: float
        :keyword noi_pr: noise prior value
            Default=1e0
        """

        # super
        super(BOTMNode, self).__init__(**kwargs)

        # members
        self._ovlp_taus = kwargs.get('ovlp_taus', None)
        if self._ovlp_taus is not None:
            self._ovlp_taus = list(self._ovlp_taus)
            if self.debug is True:
                print 'using overlap channels'
        else:
            if self.debug is True:
                print 'using subtractive interference cancelation'
        self._disc = None
        self._pr_n = None
        self._lpr_n = None
        self._pr_s = None
        self._lpr_s = None
        self._oc_idx = None
        self._debug_res = None
        self.noise_prior = kwargs.get('noi_pr', 1e0)
        self.spike_prior = kwargs.get('spk_pr', 1e-6)

    ## properties

    def get_noise_prior(self):
        return self._pr_n

    def set_noise_prior(self, value):
        if value <= 0.0:
            raise ValueError('noise prior <= 0.0')
        self._pr_n = float(value)
        self._lpr_n = sp.log(self._pr_n)

    noise_prior = property(get_noise_prior, set_noise_prior)

    def get_spike_prior(self):
        return self._pr_s

    def set_spike_prior(self, value):
        if value <= 0.0:
            raise ValueError('spike prior <= 0.0')
        self._pr_s = float(value)
        self._lpr_s = sp.log(self._pr_s)

    spike_prior = property(get_spike_prior, set_spike_prior)

    ## filter bank implementation

    def _post_filter(self):
        """build discriminant functions, prepare for sorting"""

        # tune filter outputs to prob. model
        ns = self._fout.shape[0]
        nf = self.nfilter
        if self._ovlp_taus is not None:
            nf += nf * (nf - 1) * 0.5 * len(self._ovlp_taus)
        self._disc = sp.empty((ns, nf), dtype=self.dtype)
        self._disc[:] = sp.nan
        for i in xrange(self.nfilter):
            self._disc[:, i] = (self._fout[:, i] + self._lpr_s -
                                .5 * self._xcorrs[i, i, self._tf - 1])

        # build overlap channels from filter outputs for overlap channels
        if self._ovlp_taus is not None:
            self._oc_idx = {}
            oc_idx = self.nfilter
            for f0 in xrange(self.nfilter):
                for f1 in xrange(f0 + 1, self.nfilter):
                    for tau in self._ovlp_taus:
                        self._oc_idx[oc_idx] = (f0, f1, tau)
                        f0_lim = [max(0, 0 - tau), min(ns, ns - tau)]
                        f1_lim = [max(0, 0 + tau), min(ns, ns + tau)]
                        self._disc[f0_lim[0]:f0_lim[1], oc_idx] = (
                            self._disc[f0_lim[0]:f0_lim[1], f0] +
                            self._disc[f1_lim[0]:f1_lim[1], f1] -
                            self._xcorrs[f0, f1, self._tf + tau - 1])
                        oc_idx += 1

    def _sort_chunk(self, offset):
        """sort this chunk on the calculated discriminant functions

        method: "och"
            Examples for overlap samples
                  tau=-2     tau=-1      tau=0      tau=1      tau=2
            f1:  |-----|    |-----|    |-----|    |-----|    |-----|
            f2:    |-----|   |-----|   |-----|   |-----|   |-----|
            res:    +++       ++++      +++++      ++++       +++
        method: "sic"
            TODO:
        """

        # inits
        offset = int(offset)
        spk_ep = epochs_from_binvec(
            sp.nanmax(self._disc, axis=1) > self._lpr_n)
        if spk_ep.size == 0:
            return
        min_dist = self._tf / 2
        min_size = self._tf
        for i in xrange(spk_ep.shape[0]):
            s = spk_ep[i, 1] - spk_ep[i, 0]
            if s < min_size:
                l, r = get_cut(min_size - s)
                spk_ep[i, 0] -= l
                spk_ep[i, 1] += r

        # check epochs
        spk_ep = merge_epochs(spk_ep, min_dist=min_dist)
        n_ep = spk_ep.shape[0]

        if self.debug:
            from spikeplot import mcdata

            mcdata(self._chunk, other=self._disc,
                   epochs=spk_ep - int(self._tf / 2),
                   show=False)

        for i in xrange(n_ep):
            #
            # method: overlap channels
            #
            if self._ovlp_taus is not None:
                # get event time and channel
                ep_t, ep_c = matrix_argmax(
                    self._disc[spk_ep[i, 0]:spk_ep[i, 1]])
                ep_t += spk_ep[i, 0]

                # lets fill in the results
                if ep_c < self.nfilter:
                    # was single unit
                    self.rval[ep_c].append(ep_t + offset)
                else:
                    # was overlap
                    my_oc_idx = self._oc_idx[ep_c]
                    self.rval[my_oc_idx[0]].append(ep_t + offset)
                    self.rval[my_oc_idx[1]].append(
                        ep_t + my_oc_idx[2] + offset)

            #
            # method: subtractive interference cancelation
            #
            else:
                ep_fout = self._fout[spk_ep[i, 0]:spk_ep[i, 1], :]
                ep_fout_norm = sp_la.norm(ep_fout)
                ep_disc = self._disc[spk_ep[i, 0]:spk_ep[i, 1], :].copy()

                niter = 0
                while sp.nanmax(ep_disc) > self._lpr_n:
                    # fail on spike overflow
                    niter += 1
                    if niter > self.nfilter:
                        warnings.warn(
                            'more spikes than filters found! '
                            'epoch: [%d:%d] %d' % (spk_ep[i][0] + offset,
                                                   spk_ep[i][1] + offset,
                                                   niter))
                        if niter > 2 * self.nfilter:
                            break

                    # find spike classes
                    ep_t = sp.nanargmax(sp.nanmax(ep_disc, axis=1))
                    ep_c = sp.nanargmax(ep_disc[ep_t])

                    # build subtractor
                    sub = shifted_matrix_sub(
                        sp.zeros_like(ep_disc),
                        self._xcorrs[ep_c, :, :].T,
                        ep_t - self._tf + 1)

                    # apply subtractor
                    if ep_fout_norm > sp_la.norm(ep_fout + sub):
                        if self.debug is True:
                            from spikeplot import plt, COLOURS

                            x_range = sp.arange(spk_ep[i, 0] + offset,
                                                spk_ep[i, 1] + offset)
                            f = plt.figure()
                            f.suptitle('spike epoch [%d:%d] #%d' %
                                       (spk_ep[i, 0] + offset,
                                        spk_ep[i, 1] + offset, niter))
                            ax1 = f.add_subplot(211)
                            ax1.plot(x_range, sp.zeros_like(x_range), 'k--')
                            ax1.plot(x_range, ep_disc)
                            ax1.axvline(spk_ep[i, 0] + ep_t)
                            ax2 = f.add_subplot(212, sharex=ax1, sharey=ax1)
                            ax2.plot(x_range, sub)
                            ax2.axvline(spk_ep[i, 0] + ep_t)
                        ep_disc += sub + self._lpr_s
                        if self.debug is True:
                            ax1.plot(x_range, ep_disc, ls=':', lw=2)
                        self.rval[ep_c].append(spk_ep[i, 0] + ep_t + offset)
                    else:
                        break
                del ep_fout, ep_disc, sub

    ## output methods

    def sorting2gdf(self, fname):
        """yield the gdf representing the current sorting"""

        from spikepy.common import GdfFile

        GdfFile.write_gdf(fname, self.rval)

    def plot_sorting(self, ph=None, show=False, debug=False):
        """plot the sorting of the last data chunk"""

        # imports
        from spikeplot import mcdata, COLOURS

        # create events
        ev = {}
        if self.rval is not None:
            for u in self.rval:
                ev[u] = (self.bank[u].xi, self.rval[u])

        # create colours
        cols = COLOURS[:self.nfilter]

        # calc discriminants for single units
        other = sp.empty((self._data.shape[0], self.nfilter))
        other[:] = sp.nan
        for i in xrange(self.nfilter):
            other[:, i] = self.bank[i](self._data) -\
                          0.5 * self._xcorrs[i, i, self._tf - 1] +\
                          self._lpr_s

        # plot mcdata
        mcdata(self._data, other=other, events=ev,
               plot_handle=ph, colours=cols, show=show)

##---MAIN

def main_single(do_plot=True):
    from spikeplot import plt, mcdata
    import time

    # test setup
    C_SIZE = 410
    TF = 21
    NC = 2
    CS = tuple(range(NC))
    xi1 = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi, TF))] * NC).T * 2
    xi2 = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi, TF))] * NC).T * 5
    templates = sp.asarray([xi1, xi2])
    LEN = 2000
    noise = sp.randn(LEN, NC)
    ce = TimeSeriesCovE(tf_max=TF, nc=NC)
    ce.new_chan_set(CS)
    ce.update(noise)
    FB = BOTMNode(templates=templates,
                  chan_set=CS,
                  ce=ce,
                  adapt_templates=15,
                  learn_noise=False,
                  debug=False,
                  spk_pr=1e-6,
                  ovlp_taus=None)
    signal = sp.zeros_like(noise)
    NPOS = 4
    POS = [(int(i * LEN / (NPOS + 1)), 100) for i in xrange(1, NPOS + 1)]
    POS.append((100, 2))
    POS.append((120, 2))
    print POS
    for pos, tau in POS:
        signal[pos:pos + TF] += xi1
        signal[pos + tau:pos + tau + TF] += xi2
    x = sp.ascontiguousarray(signal + noise, dtype=sp.float32)

    # sort
    tic_o = time.clock()
    FB(x)
    toc_o = time.clock()
    print 'duration:', toc_o - tic_o

    # plotting
    if do_plot:
        ev = {}
        for u in xrange(FB.nfilter):
            ev[u] = (FB.bank[u].xi, FB.rval[u])
        fouts = FB._disc
        print ev
        ovlp_meth = 'sic'
        if FB._ovlp_taus is not None:
            ovlp_meth = 'och'
        print 'overlap method:', ovlp_meth
        mcdata(x, events=ev, other=fouts,
               title='overlap method: %s' % ovlp_meth)
        FB.plot_xvft()
        plt.show()


def main_double(do_plot=True):
    from spikeplot import mcdata, plt
    import time

    # test setup
    C_SIZE = 410
    TF = 21
    NC = 2
    CS = tuple(range(NC))
    xi1 = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi, TF))] * NC).T * 2
    xi2 = sp.vstack([sp.sin(sp.linspace(0, 2 * sp.pi, TF))] * NC).T * 5
    templates = sp.asarray([xi1, xi2])
    LEN = 2000
    noise = sp.randn(LEN, NC)
    ce = TimeSeriesCovE(tf_max=TF, nc=NC)
    ce.new_chan_set(CS)
    ce.update(noise)
    FB_online = BOTMNode(templates=templates,
                         chan_set=CS,
                         ce=ce,
                         adapt_templates=15,
                         learn_noise=False,
                         debug=False)
    FB_chunked = BOTMNode(templates=templates,
                          chan_set=CS,
                          ce=ce,
                          adapt_templates=15,
                          learn_noise=False,
                          chunk_size=C_SIZE,
                          debug=False)
    signal = sp.zeros_like(noise)
    NPOS = 4
    POS = [(int(i * LEN / (NPOS + 1)), 100) for i in xrange(1, NPOS + 1)]
    POS.append((100, 2))
    print POS
    for pos, tau in POS:
        signal[pos:pos + TF] += xi1
        signal[pos + tau:pos + tau + TF] += xi2
    x = sp.ascontiguousarray(signal + noise, dtype=sp.float32)

    #    # sort chunked
    #    tic_c = time.clock()
    #    FB_chunked(x)
    #    toc_c = time.clock()
    #    print 'duration:', toc_c - tic_c
    #    if do_plot:
    #        FB_chunked.plot_sorting(show=False, debug=False)

    # sort online
    fouts = []
    rval = []
    tic_o = time.clock()
    off = 0
    while off < x.shape[0]:
        FB_online(x[off:off + C_SIZE])
        fouts.append(FB_online._disc.copy())
        rval.append((FB_online.rval.copy(), off))
        off += C_SIZE
    toc_o = time.clock()
    print 'duration:', toc_o - tic_o
    ev = {}
    for c_res, off in rval:
        for k in c_res:
            if k not in ev:
                ev[k] = []
            for t in c_res[k]:
                ev[k].append(t + off)
    dict_list_to_ndarray(ev)
    for u in ev:
        ev[u] = (FB_online.bank[u].xi, ev[u])
    fouts = sp.vstack(fouts)
    print ev
    mcdata(x, events=ev, other=fouts, title='overlap method: %s' % ovlp_meth)
    print 'overlap method:', ovlp_meth
    plt.show()

if __name__ == '__main__':
    from sys import argv
    import cProfile

    DO_PROFILE = False

    if len(argv) > 1:
        DO_PROFILE = str(argv[2]).lower() == 'true'

    if DO_PROFILE:
        cProfile.run('main_single(False)', sort=2)
    else:
        main_single(True)
