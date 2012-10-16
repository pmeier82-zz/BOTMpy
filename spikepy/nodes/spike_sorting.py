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

"""implementation of spike sorting with optimal linear filters

See:
[1] F. Franke, M. Natora, C. Boucsein, M. Munk, and K. Obermayer. An online
spike detection and spike classification algorithm capable of instantaneous
resolution of overlapping spikes. Journal of Computational Neuroscience, 2009
[2] F. Franke, ... , 2012,
The revolutionary BOTM Paper
"""

__docformat__ = 'restructuredtext'
__all__ = ['FilterBankSortingNode', 'AdaptiveBayesOptimalTemplateMatchingNode',
           'BayesOptimalTemplateMatchingNode', 'BOTMNode', 'ABOTMNode']

##---IMPORTS

import copy
import scipy as sp
from scipy import linalg as sp_la

try:
    from sklearn.mixture import log_multivariate_normal_density
except ImportError:
    from sklearn.mixture import lmvnpdf as log_multivariate_normal_density
from sklearn.utils.extmath import logsumexp
import warnings
from .base_nodes import PCANode
from .cluster import HomoscedasticClusteringNode
from .filter_bank import FilterBankError, FilterBankNode
from .prewhiten import PrewhiteningNode2
from .spike_detection import SDMteoNode, ThresholdDetectorNode
from ..common import (
    overlaps, epochs_from_spiketrain, epochs_from_spiketrain_set, shifted_matrix_sub, mcvec_to_conc,
    epochs_from_binvec, merge_epochs, matrix_argmax, dict_list_to_ndarray, get_cut, GdfFile,
    MxRingBuffer, mcvec_from_conc, extract_spikes)

##---CONSTANTS

MTEO_DET = SDMteoNode
MTEO_PARAMS = tuple(), {'kvalues': [3, 9, 15, 21],
                        'threshold_factor': 0.98,
                        'min_dist': 32}

##---CLASSES

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
        :type templates: ndarray
        :keyword templates: templates to initialise the filter stack.
            [ntemps][tf][nc] a tensor of templates
            Required
        :type ce: TimeSeriesCovE
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
                self.create_filter(temp)

    ## SortingNode interface

    def _execute(self, x):
        # init
        self._data = x[:, self._chan_set]
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
            self._chunk = self._data[self._chunk_offset:self._chunk_offset + clen]
            self._fout = sp.empty((clen, self.nf))

            # filtering
            self._pre_filter()
            self._fout = super(FilterBankSortingNode, self)._execute(self._chunk)
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

    def spikes_u(self, u, mc=True, exclude_overlaps=True, overlap_window=None):
        """yields the spike for the u-th filter

        :type u: int
        :param u: index of the filter # CHECK THIS
        :type mc: bool
        :param mc: if True, return spikes multi-channeled, else return spikes concatenated
            Default=True
        :type exclude_overlaps: bool
        :param exclude_overlaps: if True, exclude overlap spike
        :type overlap_window: int
        :param overlap_window: if `exclude_overlaps` is True, this will define the overlap range,
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
        if u not in self.rval:
            rval = sp.zeros(size)
        else:
            ep, st_dict[u] = epochs_from_spiketrain(
                st_dict[u], self._tf, end=self._data.shape[0], with_corrected_st=True)
            if ep.size == 0:
                rval = sp.zeros(size)
            else:
                rval = extract_spikes(self._data, ep, mc=mc)
                if exclude_overlaps is True:
                    ovlp_info = overlaps(st_dict, overlap_window or self._tf)[0]
                    rval = rval[ovlp_info[u] == False]

        return rval


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
        if self._data is None or self.rval is None or len(self._idx_active_set) == 0:
            warnings.warn('not initialised properly to plot a sorting!')
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
            other -= [.5 * self.get_xcorrs_at(i)
                      for i in xrange(self.nf)]

        # plot mcdata
        return mcdata(self._data, other=other, events=ev,
            plot_handle=ph, colours=cols, show=show)

    def plot_sorting_waveforms(self, ph=None, show=False):
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
        if self._data is None or self.rval is None or len(self._idx_active_set) == 0:
            warnings.warn('not initialised properly to plot a sorting!')
            return None

        # inits
        wf = {}
        temps = {}
        #if self._data is None or self.rval is None:
        #    return
        cut = get_cut(self._tf)


        # adapt filters with found waveforms
        nunits = 0
        for u in self.rval:
            spks_u = self.spikes_u(u, mc=False)
            if spks_u.size > 0:
                wf[u] = self.spikes_u(u)
                temps[u] = self.bank[u].xi_conc
                nunits += 1
        print 'waveforms for units:', nunits

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


class BayesOptimalTemplateMatchingNode(FilterBankSortingNode):
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

        #kwargs
        ovlp_taus = kwargs.pop('ovlp_taus', None)
        noi_pr = kwargs.pop('noi_pr', 1e0)
        spk_pr = kwargs.pop('spk_pr', 1e-6)

        # super
        super(BayesOptimalTemplateMatchingNode, self).__init__(**kwargs)

        # members
        self._ovlp_taus = ovlp_taus
        if self._ovlp_taus is not None:
            self._ovlp_taus = list(self._ovlp_taus)
            if self.verbose.has_print:
                print 'using overlap channels'
        else:
            if self.verbose.has_print:
                print 'using subtractive interference cancelation'
        self._disc = None
        self._pr_n = None
        self._lpr_n = None
        self._pr_s = None
        self._lpr_s = None
        self._oc_idx = None
        self.noise_prior = noi_pr
        self.spike_prior = spk_pr

    ## properties

    def get_noise_prior(self):
        return self._pr_n

    def set_noise_prior(self, value):
        if value <= 0.0:
            raise ValueError('noise prior <= 0.0')
        if value > 1.0:
            raise ValueError('noise prior > 1.0')
        self._pr_n = float(value)
        self._lpr_n = sp.log(self._pr_n)

    noise_prior = property(get_noise_prior, set_noise_prior)

    def get_spike_prior(self):
        return self._pr_s

    def set_spike_prior(self, value):
        if value <= 0.0:
            raise ValueError('spike prior <= 0.0')
        if value > 1.0:
            raise ValueError('spike prior > 1.0')
        self._pr_s = float(value)
        self._lpr_s = sp.log(self._pr_s)

    spike_prior = property(get_spike_prior, set_spike_prior)

    ## filter bank sorting interface

    def _post_filter(self):
        """build discriminant functions, prepare for sorting"""

        # tune filter outputs to prob. model
        ns = self._fout.shape[0]
        nf = self.nf
        if self._ovlp_taus is not None:
            nf += nf * (nf - 1) * 0.5 * len(self._ovlp_taus)
        self._disc = sp.empty((ns, nf), dtype=self.dtype)
        self._disc[:] = sp.nan
        for i in xrange(self.nf):
            self._disc[:, i] = (self._fout[:, i] + self._lpr_s -
                                .5 * self.get_xcorrs_at(i))

        # build overlap channels from filter outputs for overlap channels
        if self._ovlp_taus is not None:
            self._oc_idx = {}
            oc_idx = self.nf
            for f0 in xrange(self.nf):
                for f1 in xrange(f0 + 1, self.nf):
                    for tau in self._ovlp_taus:
                        self._oc_idx[oc_idx] = (f0, f1, tau)
                        f0_lim = [max(0, 0 - tau), min(ns, ns - tau)]
                        f1_lim = [max(0, 0 + tau), min(ns, ns + tau)]
                        self._disc[f0_lim[0]:f0_lim[1], oc_idx] = (
                            self._disc[f0_lim[0]:f0_lim[1], f0] +
                            self._disc[f1_lim[0]:f1_lim[1], f1] -
                            self.get_xcorrs_at(f0, f1, tau))
                        oc_idx += 1

    def _sort_chunk(self):
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
        if self.nf == 0:
            return
        spk_ep = epochs_from_binvec(
            sp.nanmax(self._disc, axis=1) > self._lpr_n)
        if spk_ep.size == 0:
            return
        l, r = get_cut(self._tf)
        for i in xrange(spk_ep.shape[0]):
            # FIX: for now we just continue for empty epochs, where do they come from anyways?!
            if spk_ep[i, 1] - spk_ep[i, 0] < 1:
                continue
            mc = self._disc[spk_ep[i, 0]:spk_ep[i, 1], :].argmax(0).argmax()
            s = self._disc[spk_ep[i, 0]:spk_ep[i, 1], mc].argmax() + spk_ep[i, 0]
            spk_ep[i] = [s - l, s + r]

        # check epochs
        spk_ep = merge_epochs(spk_ep)
        n_ep = spk_ep.shape[0]

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
                if ep_c < self.nf:
                    # was single unit
                    fid = self.get_idx_for(ep_c)
                    self.rval[fid].append(ep_t + self._chunk_offset)
                else:
                    # was overlap
                    my_oc_idx = self._oc_idx[ep_c]
                    fid0 = self.get_idx_for(my_oc_idx[0])
                    self.rval[fid0].append(ep_t + self._chunk_offset)
                    fid1 = self.get_idx_for(my_oc_idx[1])
                    self.rval[fid1].append(
                        ep_t + my_oc_idx[2] + self._chunk_offset)

            #
            # method: subtractive interference cancelation
            #
            else:
                ep_fout = self._fout[spk_ep[i, 0]:spk_ep[i, 1], :]
                ep_fout_norm = sp_la.norm(ep_fout)
                ep_disc = self._disc[spk_ep[i, 0]:spk_ep[i, 1], :].copy()

                niter = 0
                while sp.nanmax(ep_disc) > self._lpr_n:
                    # warn on spike overflow
                    niter += 1
                    if niter > self.nf:
                        warnings.warn(
                            'more spikes than filters found! '
                            'epoch: [%d:%d] %d' % (
                                spk_ep[i][0] + self._chunk_offset,
                                spk_ep[i][1] + self._chunk_offset,
                                niter))
                        if niter > 2 * self.nf:
                            break

                    # find spike classes
                    ep_t = sp.nanargmax(sp.nanmax(ep_disc, axis=1))
                    ep_c = sp.nanargmax(ep_disc[ep_t])

                    # build subtrahend
                    sub = shifted_matrix_sub(
                        sp.zeros_like(ep_disc),
                        self._xcorrs[ep_c, :, :].T,
                        ep_t - self._tf + 1)

                    # apply subtrahend
                    if ep_fout_norm > sp_la.norm(ep_fout + sub):
                        if self.verbose.has_plot:
                            try:
                                from spikeplot import xvf_tensor, plt, COLOURS

                                x_range = sp.arange(
                                    spk_ep[i, 0] + self._chunk_offset,
                                    spk_ep[i, 1] + self._chunk_offset)
                                f = plt.figure()
                                f.suptitle('spike epoch [%d:%d] #%d' %
                                           (spk_ep[i, 0] + self._chunk_offset,
                                            spk_ep[i, 1] + self._chunk_offset,
                                            niter))
                                ax1 = f.add_subplot(211)
                                ax1.set_color_cycle(['k'] + COLOURS[:self.nf] * 2)
                                ax1.plot(x_range, sp.zeros_like(x_range), ls='--')
                                ax1.plot(x_range, ep_disc, label='pre_sub')
                                ax1.axvline(x_range[ep_t], c='k')
                                ax2 = f.add_subplot(212, sharex=ax1, sharey=ax1)
                                ax2.set_color_cycle(['k'] + COLOURS[:self.nf])
                                ax2.plot(x_range, sp.zeros_like(x_range), ls='--')
                                ax2.plot(x_range, sub)
                                ax2.axvline(x_range[ep_t], c='k')
                            except ImportError:
                                pass
                        ep_disc += sub + self._lpr_s
                        if self.verbose.has_plot:
                            try:
                                ax1.plot(x_range, ep_disc, ls=':', lw=2, label='post_sub')
                                ax1.legend(loc=2)
                            except ImportError:
                                pass
                        fid = self.get_idx_for(ep_c)
                        self.rval[fid].append(
                            spk_ep[i, 0] + ep_t + self._chunk_offset)
                    else:
                        break
                del ep_fout, ep_disc, sub

    ## BOTM implementation

    def posterior_prob(self, obs, with_noise=False):
        """posterior probabilities for data under the model

        :type obs: ndarray
        :param obs: observations to be evaluated [n, tf, nc]
        :type with_noise: bool
        :param with_noise: if True, include the noise cluster as component
            in the mixture.
            Default=False
        :rtype: ndarray
        :returns: matrix with per component posterior probabilities [n, c]
        """

        # check obs
        obs = sp.atleast_2d(obs)
        if len(obs) == 0:
            raise ValueError('no observations passed!')
        data = []
        if obs.ndim == 2:
            if obs.shape[1] != self._tf * self._nc:
                raise ValueError('data dimensions not compatible with model')
            for i in xrange(obs.shape[0]):
                data.append(obs[i])
        elif obs.ndim == 3:
            if obs.shape[1:] != (self._tf, self._nc):
                raise ValueError('data dimensions not compatible with model')
            for i in xrange(obs.shape[0]):
                data.append(mcvec_to_conc(obs[i]))
        data = sp.asarray(data, dtype=sp.float64)

        # build comps
        comps = self.get_template_set(mc=False)
        if with_noise:
            comps = sp.vstack((comps, sp.zeros((self._tf * self._nc))))
        comps = comps.astype(sp.float64)
        if len(comps) == 0:
            return sp.zeros((len(obs), 1))

        # build priors
        prior = sp.array([self._lpr_s] * len(comps), dtype=sp.float64)
        if with_noise:
            prior[-1] = self._lpr_n

        # get sigma
        try:
            sigma = self._ce.get_cmx(tf=self._tf).astype(sp.float64)
        except:
            return sp.zeros((len(obs), 1))

        # calc log probs
        lpr = log_multivariate_normal_density(data, comps, sigma, 'tied') + prior
        logprob = logsumexp(lpr, axis=1)
        return sp.exp(lpr - logprob[:, sp.newaxis])

    def component_divergence(self, obs, with_noise=False, loading=False, subdim=None):
        """component probabilities under the model

        :type obs: ndarray
        :param obs: observations to be evaluated [n, tf, nc]
        :type with_noise: bool
        :param with_noise: if True, include the noise cluster as component
            in the mixture.
            Default=False
        :type loading: bool
        :param loading: if True, use the loaded matrix
            Default=False
        :type subdim: int
        :param subdim: dimensionality of subspace to build the inverse over.
            if None ignore
            Default=None
        :rtype: ndarray
        :returns: divergence from means of current filter bank[n, c]
        """

        # check data
        obs = sp.atleast_2d(obs)
        if len(obs) == 0:
            raise ValueError('no observations passed!')
        data = []
        if obs.ndim == 2:
            if obs.shape[1] != self._tf * self._nc:
                raise ValueError('data dimensions not compatible with model')
            for i in xrange(obs.shape[0]):
                data.append(obs[i])
        elif obs.ndim == 3:
            if obs.shape[1:] != (self._tf, self._nc):
                raise ValueError('data dimensions not compatible with model')
            for i in xrange(obs.shape[0]):
                data.append(mcvec_to_conc(obs[i]))
        data = sp.asarray(data, dtype=sp.float64)

        # build component
        comps = self.get_template_set(mc=False)
        if with_noise:
            comps = sp.vstack((comps, sp.zeros((self._tf * self._nc))))
        comps = comps.astype(sp.float64)
        if len(comps) == 0:
            return sp.ones((len(obs), 1)) * sp.inf

        # get sigma
        try:
            if loading is True:
                sigma_inv = self._ce.get_icmx_loaded(tf=self._tf).astype(
                    sp.float64)
            else:
                sigma_inv = self._ce.get_icmx(tf=self._tf).astype(sp.float64)
            if subdim is not None:
                subdim = int(subdim)
                svd = self._ce.get_svd(tf=self._tf).astype(sp.float64)
                sv = svd[1].copy()
                t = sp.finfo(self._ce.dtype).eps * len(sv) * svd[1].max()
                sv[sv < t] = 0.0
                sigma_inv = sp.dot(svd[0][:, :subdim],
                    sp.dot(sp.diag(1. / sv[:subdim]),
                        svd[2][:subdim]))
        except:
            return sp.ones((len(obs), 1)) * sp.inf

        # return component wise divergence
        rval = sp.zeros((obs.shape[0], comps.shape[0]), dtype=sp.float64)
        for n in xrange(obs.shape[0]):
            x = data[n] - comps
            for c in xrange(comps.shape[0]):
                rval[n, c] = sp.dot(x[c], sp.dot(sigma_inv, x[c]))
        return rval

# for legacy compatibility
BOTMNode = BayesOptimalTemplateMatchingNode

class AdaptiveBayesOptimalTemplateMatchingNode(BayesOptimalTemplateMatchingNode):
    """Adaptive BOTM Node

    tries to match parallel detection with sorting to find new units.
    """

    def __init__(self, **kwargs):
        """
        :type learn_templates: int
        :keyword learn_templates: if non-negative integer, adapt the filters
            with the found events aligned at that sample. If negative,
            calculate the alignment samples as int(.25*self.tf)
            Default=-1
        :type learn_noise: str or None
        :keyword learn_noise: if not None, adapt the noise covariance matrix
            with from the noise epochs. This has to be either 'sort' to
            learn from the non overlapping sorting events,
            or 'det' to lean from the detection. Else, do not learn the noise.
            Default='sort'
        :type det_cls: ThresholdDetectorNode
        :keyword det_cls: the class of detector node to use for the spike
            detection running in parallel to the sorting,
            this must be a subclass of 'ThresholdDetectorNode'.
            Default=MTEO_DET
        :type det_limit: int
        :keyword det_limit: capacity of the ringbuffer to hold the detection
            spikes.
            Default=2000
        :type det_params: dict
        :keyword det_params: parameters for the spike detector that will be
            run in parallel on the data.
            Default=MTEO_PARAMS
        """

        # kwargs
        learn_templates_pval = kwargs.pop('learn_templates_pval', 0.05)
        learn_templates = kwargs.pop('learn_templates', -1)
        learn_noise = kwargs.pop('learn_noise', None)
        det_cls = kwargs.pop('det_cls', MTEO_DET)
        det_limit = kwargs.pop('det_limit', 2000)
        det_params = kwargs.pop('det_params', MTEO_PARAMS)

        # check det_cls
        if not issubclass(det_cls, ThresholdDetectorNode):
            raise TypeError(
                '\'det_cls\' of type ThresholdDetectorNode is required!')
        if learn_noise is not None:
            if learn_noise not in ['det', 'sort']:
                learn_noise = None

        # super
        super(AdaptiveBayesOptimalTemplateMatchingNode, self).__init__(**kwargs)

        if learn_templates < 0:
            learn_templates = int(0.25 * self._tf)

        # members
        self._det = None
        self._det_cls = det_cls
        self._det_params = det_params
        self._det_limit = int(det_limit)
        self._det_buf = None
        self._learn_noise = learn_noise
        self._learn_templates = learn_templates
        self._learn_templates_pval = learn_templates_pval

        # align at (learn_templates)
        if self._learn_templates < 0:
            self._learn_templates = .25
        if isinstance(self._learn_templates, float):
            if 0.0 <= self._learn_templates <= 1.0:
                self._learn_templates *= self.tf
            self._learn_templates = int(self._learn_templates)

    ## properties

    def get_det(self):
        if self._det is None:
            self._det = self._det_cls(tf=self._tf, *self._det_params[0],
                **self._det_params[1])
            self._det_buf = MxRingBuffer(capacity=self._det_limit,
                dimension=(self._tf * self._nc),
                dtype=self.dtype)
            if self.verbose.has_print:
                print 'build detector:', self._det_cls, self._det_params
        return self._det

    det = property(get_det)

    ## filter bank sorting interface

    def _event_explained(self, ev, win_half_span=15):
        """check event for explanation by the filter bank"""

        if not self._disc.size:
            return False
        cut = self._learn_templates, self.tf - self._learn_templates
        disc_at = ev + cut[1] - 1
        if self.verbose.has_plot:
            try:
                from spikeplot import  mcdata

                at = disc_at - win_half_span, disc_at + win_half_span
                evts = {0: [ev - at[0]], 1: [disc_at - at[0]]}
                mcdata(
                    data=self._chunk[at[0] - self.tf:at[1]], other=self._disc[at[0]:at[1]], events=evts,
                    x_offset=at[0],
                    title='det@%s(%s) disc@%s' % (ev, self._learn_templates, disc_at), show=False)
            except ImportError:
                pass
        return self._disc[disc_at - win_half_span:disc_at + win_half_span, :].max() >= 0.0

    def _post_sort(self):
        self.det.reset()
        self.det(self._chunk)
        print '_post_sort, self._learn_templates:', self._learn_templates
        spks = self.det.get_extracted_events(mc=False, kind='min', align_at=self._learn_templates)
        spks_explained = sp.array([self._event_explained(e) for e in self.det.events])
        if len(spks[spks_explained == False]) > 0:
            self._det_buf.extend(spks[spks_explained == False])

    def _execute(self, x):
        # call super to get sorting
        rval = super(AdaptiveBayesOptimalTemplateMatchingNode, self)._execute(x)
        # adaption of noise covariance
        self._adapt_noise()
        # adaption filter bank
        self._adapt_filter_drop()
        self._adapt_filter_current()
        self._adapt_filter_new()
        # learn slow noise statistic changes
        return rval

    ## ABOTM interface

    def _adapt_filter_drop(self):
        nsmpl = self._data.shape[0]
        for u in list(self._idx_active_set):
            # 1) snr drop below 0.5
            if self.bank[u].get_snr < 0.5:
                self.deactivate(u)
                warnings.warn('deactivating filter %s, snr' % str(u))

            # 2) rate drop below 0.1
            if hasattr(self.bank[u], 'rate'):
                try:
                    nspks = len(self.rval[u])
                except:
                    nspks = 0
                self.bank[u].rate.observation(nspks, nsmpl)
                if self.bank[u].rate.estimate() < 1.0:
                    if self.bank[u].rate._n_updates_since == 10:
                        self.deactivate(u)
                        warnings.warn('deactivating filter %s, rate' % str(u))
        self._check_internals()

    def _adapt_filter_new(self):
        if self._det_buf.is_full:
            if self.verbose.has_print:
                print 'det_buf is full!'

            # get all spikes and clear buffer
            spks = self._det_buf[:].copy()
            self._det_buf.clear()

            # processing chain
            pre_pro = PrewhiteningNode2(self._ce) + PCANode(output_dim=10)
            clus = HomoscedasticClusteringNode(
                clus_type='dpgmm',
                cvtype='tied',
                debug=self.verbose.has_print,
                alpha=16.0,
                crange=[20],
                max_iter=2096)
            spks_pp = pre_pro(spks)
            clus(spks_pp)
            lbls = clus.labels
            if self.verbose.has_plot:
                clus.plot(spks_pp, show=True)
            for i in sp.unique(lbls):
                if self.verbose.has_print:
                    print 'checking new unit:',
                spks_i = spks[lbls == i]
                # TODO: parametrise this
                if len(spks_i) < 50:
                    self._det_buf.extend(spks_i)
                    if self.verbose.has_print:
                        print 'rejected, only %d spikes' % len(spks_i)
                else:
                    spk_i = mcvec_from_conc(spks_i.mean(0), nc=self._nc)
                    self.create_filter(spk_i)
                    if self.verbose.has_print:
                        print 'accepted, with %d spikes' % len(spks_i)
            del pre_pro, clus, spks, spks_pp
        else:
            if self.verbose.has_print:
                print 'det_buf:', self._det_buf

    def _adapt_noise(self):
        if self._learn_noise:
            nep = None
            if self._learn_noise == 'sort':
                if len(self.rval) > 0:
                    nep = epochs_from_spiketrain_set(
                        self.rval, cut=self._tf,
                        end=self._data.shape[0])['noise']
            elif self._learn_noise == 'det':
                if len(self.det.events) > 0:
                    nep = self.det.get_epochs(merge=True, invert=True)
            else:
                raise ValueError('unrecognised value for learn_noise: %s' % str(self._learn_noise))
            self._ce.update(self._data, epochs=nep)

    def _adapt_filter_current(self):
        """adapt templates/filters using non overlapping spikes"""

        # check and init
        if self._data is None or self.rval is None:
            return

        # adapt filters with found waveforms
        for u in self.rval:
            spks_u = self.spikes_u(u, mc=True, exclude_overlaps=True)
            if spks_u.size == 0:
                continue
            self.bank[u].extend_xi_buf(spks_u)
            self.bank[u].rate.observation(spks_u.shape[0], self._data.shape[0])
        print [(u, f.rate.estimate()) for (u, f) in self.bank.items()]

ABOTMNode = AdaptiveBayesOptimalTemplateMatchingNode

##---MAIN

if __name__ == '__main__':
    pass
