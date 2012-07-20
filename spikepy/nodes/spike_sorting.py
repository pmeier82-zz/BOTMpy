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
[2] F. Franke, ... , K. Obermayer, 2012,
The revolutionary BOTM Paper
"""

__docformat__ = 'restructuredtext'
__all__ = ['ABOTMNode', 'BOTMNode', 'BayesOptimalTemplateMatchingNode',
           'FilterBankSortingNode']

##---IMPORTS

import scipy as sp
from scipy import linalg as sp_la
from sklearn.mixture import lmvnpdf
from sklearn.utils.extmath import logsumexp
from spikeplot import COLOURS, mcdata, plt, waveforms
import warnings
from .base_nodes import Node, PCANode
from .cluster import HomoscedasticClusteringNode
from .filter_bank import FilterBankError, FilterBankNode
from .prewhiten import PrewhiteningNode2
from .spike_detection import SDMteoNode, ThresholdDetectorNode
from ..common import (
    overlaps, epochs_from_spiketrain_set, shifted_matrix_sub, mcvec_to_conc,
    epochs_from_binvec, merge_epochs, matrix_argmax, dict_list_to_ndarray,
    get_cut, get_aligned_spikes, GdfFile, MxRingBuffer, mcvec_from_conc,
    VERBOSE)

##---CONSTANTS

MTEO_DET = SDMteoNode
#MTEO_PARAMS = tuple(), {'kvalues':[6, 9, 13, 18],
MTEO_PARAMS = tuple(), {'kvalues': [1, 3, 5, 7, 9],
                        'threshold_factor': 3.0,
                        'min_dist': 5}

##---CLASSES

class FilterBankSortingNode(Node):
    """abstract class that handles filter instances and their outputs

    This class provides a pipeline structure to implement spike sorting
    algorithms that operate on a filter bank. The implementation is done by
    implementing `self._pre_filter`, `self._post_filter`, `self._sort_chunk`
    and `self._post_sort` methods with meaning full processing. After the
    filter steps the filter output of the filters constituting the filter
    bank is present and can be processed on. Input data can be partitioned
    into chunks of smaller size.
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
        dtype = kwargs.pop('dtype', sp.float32)
        verbose = kwargs.get('verbose', 0)
        # everything not popped goes to FilterBankNode

        # build filter bank
        bank = FilterBankNode(dtype=dtype, **kwargs)
        keywords = ['ce', 'chan_set', 'filter_cls', 'rb_cap', 'tf', 'verbose']
        for key in keywords:
            kwargs.pop(key, None)

        # everything not popped goes to super

        # super
        super(FilterBankSortingNode, self).__init__(dtype=dtype, **kwargs)

        # members
        self._bank = bank
        self._fout = None
        self._data = None
        self._chunk = None
        self._chunk_offset = 0
        self._chunk_size = int(chunk_size)
        self.verbose = VERBOSE(verbose)
        self.rval = {}

        # create filters for templates
        if templates is not None:
            for temp in templates:
                self.create_filter(temp)

    ## properties

    def get_tf(self):
        return self._bank.tf

    tf = property(get_tf)

    def get_nc(self):
        return self._bank.nc

    nc = property(get_nc)

    def get_chan_set(self):
        return self._bank.cs

    cs = property(get_chan_set)

    def get_ce(self):
        return self._bank.ce

    def set_ce(self, value):
        self._bank.ce = value

    ce = property(get_ce, set_ce)

    def get_filter_idx(self):
        return self._bank._idx_active_set

    filter_idx = property(get_filter_idx)

    def get_nfilter(self):
        return len(self.filter_idx)

    nfilter = property(get_nfilter)

    def get_template_set(self, **kwargs):
        return self._bank.get_template_set(**kwargs)

    template_set = property(get_template_set)

    def get_filter_set(self, **kwargs):
        return self._bank.get_filter_set(**kwargs)

    filter_set = property(get_filter_set)

    ## filter bank interface

    def reset_history(self):
        """sets the history to all zeros for all filters"""

        self._bank.reset_history()

    def create_filter(self, xi, check=True):
        """adds a new filter to the filter bank"""

        if xi.ndim == 1 and xi.size == self.tf * self.nc:
            xi = mcvec_from_conc(xi, nc=self.nc)
        elif xi.ndim == 2 and xi.shape == (self.tf, self.nc):
            pass
        else:
            raise ValueError('dimension missmatch! tf=%s, nc=%s' % (self.tf, self.nc))
        return self._bank.create_filter(xi, check=check)

    ## mpd.Node interface

    def is_invertible(self):
        return False

    def is_trainable(self):
        return False

    ## SortingNode interface

    def _execute(self, x):
        # inits
        self._data = x[:, self._bank.cs]
        dlen = self._data.shape[0]
        self.rval.clear()
        for i in self.filter_idx:
            self.rval[i] = []
        curr_chunk = 0
        has_next_chunk = True

        # sort per chunk
        while has_next_chunk:
            # get chunk limits
            #c_start = curr_chunk * self._chunk_size
            self._chunk_offset = curr_chunk * self._chunk_size
            clen = min(dlen, (curr_chunk + 1) * self._chunk_size)
            clen -= self._chunk_offset

            # generate data chunk and process
            self._chunk = self._data[
                          self._chunk_offset:self._chunk_offset + clen]
            self._fout = sp.empty((clen, self.nfilter))

            # filtering
            self._pre_filter()
            self._fout = self._bank(self._chunk)
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

    ## FilterBankSorting interface

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
        correct = int(self.tf / 2)
        for k in self.rval:
            self.rval[k] -= correct

    ## output methods

    def plot_xvft(self, ph=None, show=False):
        """plot the Xi vs F Tensor of the filter bank"""

        return self._bank.plot_xvft(ph=ph, show=show)

    def plot_template_set(self, ph=None, show=False):
        """plot the template set in a waveform plot"""

        return self._bank.plot_template_set(ph=ph, show=show)

    def plot_sorting(self, ph=None, show=False):
        """plot the sorting of the last data chunk"""

        # create events
        ev = {}
        if self.rval is not None:
            temps = self.template_set
            for i, k in enumerate(self.filter_idx):
                if self.rval[k].any():
                    ev[k] = (temps[i], self.rval[k])

        # create colours
        cols = COLOURS[:self.nfilter]

        # calc discriminants for single units
        other = None
        if self.nfilter > 0:
            self.reset_history()
            other = self._bank(self._data)
            other += self._lpr_s
            other -= [.5 * self._bank.get_xcorrs_at(i)
                      for i in xrange(self.nfilter)]

        # plot mcdata
        return mcdata(self._data, other=other, events=ev,
            plot_handle=ph, colours=cols, show=show)

    def plot_sorting_waveforms(self, ph=None, show=False):
        """plot the waveforms of the sorting of the last data chunk"""

        # inits
        wf = {}
        temps = {}
        #if self._data is None or self.rval is None:
        #    return
        cut = get_cut(self.tf)

        # adapt filters with found waveforms
        nunits = 0
        for u in self.rval:
            spks_u = get_aligned_spikes(self._data, self.rval[u], self.tf,
                mc=False)[0]
            if spks_u.size > 0:
                wf[u] = spks_u
                temps[u] = self._bank.bank[u].xi_conc
                nunits += 1
        print 'waveforms for units:', nunits

        """
        waveforms(waveforms, samples_per_second=None, tf=None, plot_mean=False,
              plot_single_waveforms=True, set_y_range=False,
              plot_separate=True, plot_handle=None, colours=None, title=None,
              filename=None, show=True):
        """
        return waveforms(wf, samples_per_second=None, tf=self.tf,
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
            if self.verbose.has_print():
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
        self._debug_res = None
        self.noise_prior = noi_pr
        self.spike_prior = spk_pr

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
                                .5 * self._bank.get_xcorrs_at(i))

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
                            self._bank.get_xcorrs_at(f0, f1, tau))
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
        if self.nfilter == 0:
            return
        spk_ep = epochs_from_binvec(
            sp.nanmax(self._disc, axis=1) > self._lpr_n)
        if spk_ep.size == 0:
            return
        min_dist = self.tf / 2
        min_size = int(self.tf * 1.2)
        for i in xrange(spk_ep.shape[0]):
            s = spk_ep[i, 1] - spk_ep[i, 0]
            if s < min_size:
                l, r = get_cut(min_size - s)
                spk_ep[i, 0] -= l
                spk_ep[i, 1] += r

        # check epochs
        spk_ep = merge_epochs(spk_ep, min_dist=min_dist)
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
                if ep_c < self.nfilter:
                    # was single unit
                    fid = self._bank.get_fid_for(ep_c)
                    self.rval[fid].append(ep_t + offset)
                else:
                    # was overlap
                    my_oc_idx = self._oc_idx[ep_c]
                    fid0 = self._bank.get_fid_for(my_oc_idx[0])
                    self.rval[fid0].append(ep_t + offset)
                    fid1 = self._bank.get_fid_for(my_oc_idx[1])
                    self.rval[fid1].append(
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
                            'epoch: [%d:%d] %d' % (
                                spk_ep[i][0] + self._chunk_offset,
                                spk_ep[i][1] + self._chunk_offset,
                                niter))
                        if niter > 2 * self.nfilter:
                            break

                    # find spike classes
                    ep_t = sp.nanargmax(sp.nanmax(ep_disc, axis=1))
                    ep_c = sp.nanargmax(ep_disc[ep_t])

                    # build subtrahend
                    sub = shifted_matrix_sub(
                        sp.zeros_like(ep_disc),
                        self._bank.xcorrs[ep_c, :, :].T,
                        ep_t - self.tf + 1)

                    # apply subtrahend
                    if ep_fout_norm > sp_la.norm(ep_fout + sub):
                        if self.verbose.has_plot:
                            x_range = sp.arange(
                                spk_ep[i, 0] + self._chunk_offset,
                                spk_ep[i, 1] + self._chunk_offset)
                            f = plt.figure()
                            f.suptitle('spike epoch [%d:%d] #%d' %
                                       (spk_ep[i, 0] + self._chunk_offset,
                                        spk_ep[i, 1] + self._chunk_offset,
                                        niter))
                            ax1 = f.add_subplot(211)
                            ax1.plot(x_range, sp.zeros_like(x_range), 'k--')
                            ax1.plot(x_range, ep_disc)
                            ax1.axvline(x_range[ep_t])
                            ax2 = f.add_subplot(212, sharex=ax1, sharey=ax1)
                            ax2.plot(x_range, sub)
                            ax2.axvline(x_range[ep_t])
                        ep_disc += sub + self._lpr_s
                        if self.verbose.has_plot:
                            ax1.plot(x_range, ep_disc, ls=':', lw=2)
                        fid = self._bank.get_fid_for(ep_c)
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
            if obs.shape[1] != self.tf * self.nc:
                raise ValueError('data dimensions not compatible with model')
            for i in xrange(obs.shape[0]):
                data.append(obs[i])
        elif obs.ndim == 3:
            if obs.shape[1:] != (self.tf, self.nc):
                raise ValueError('data dimensions not compatible with model')
            for i in xrange(obs.shape[0]):
                data.append(mcvec_to_conc(obs[i]))
        data = sp.asarray(data, dtype=sp.float64)

        # build comps
        comps = self.get_template_set(mc=False)
        if with_noise:
            comps = sp.vstack((comps, sp.zeros((self.tf * self.nc))))
        comps = comps.astype(sp.float64)
        if len(comps) == 0:
            return sp.zeros((len(obs), 1))

        # build priors
        prior = sp.array([self._lpr_s] * len(comps), dtype=sp.float64)
        if with_noise:
            prior[-1] = self._lpr_n

        # get sigma
        try:
            sigma = self.ce.get_cmx(tf=self.tf).astype(sp.float64)
        except:
            return sp.zeros((len(obs), 1))

        # calc log probs
        lpr = lmvnpdf(data, comps, sigma, 'tied') + prior
        logprob = logsumexp(lpr, axis=1)
        return sp.exp(lpr - logprob[:, sp.newaxis])

    def component_divergence(self, obs, with_noise=False,
                             loading=False, subdim=None):
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
            if obs.shape[1] != self.tf * self.nc:
                raise ValueError('data dimensions not compatible with model')
            for i in xrange(obs.shape[0]):
                data.append(obs[i])
        elif obs.ndim == 3:
            if obs.shape[1:] != (self.tf, self.nc):
                raise ValueError('data dimensions not compatible with model')
            for i in xrange(obs.shape[0]):
                data.append(mcvec_to_conc(obs[i]))
        data = sp.asarray(data, dtype=sp.float64)

        # build component
        comps = self.get_template_set(mc=False)
        if with_noise:
            comps = sp.vstack((comps, sp.zeros((self.tf * self.nc))))
        comps = comps.astype(sp.float64)
        if len(comps) == 0:
            return sp.ones((len(obs), 1)) * sp.inf

        # get sigma
        try:
            if loading is True:
                sigma_inv = self.ce.get_icmx_loaded(tf=self.tf).astype(
                    sp.float64)
            else:
                sigma_inv = self.ce.get_icmx(tf=self.tf).astype(sp.float64)
            if subdim is not None:
                subdim = int(subdim)
                svd = self.ce.get_svd(tf=self.tf).astype(sp.float64)
                sv = svd[1].copy()
                t = sp.finfo(self.ce.dtype).eps * len(sv) * svd[1].max()
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

class ABOTMNode(BayesOptimalTemplateMatchingNode):
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
        learn_noise = kwargs.pop('learn_noise', 'det')
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
        super(ABOTMNode, self).__init__(**kwargs)

        if learn_templates < 0:
            learn_templates = int(0.25 * self.tf)

        # members
        self._det = None
        self._det_cls = det_cls
        self._det_params = det_params
        self._det_limit = int(det_limit)
        self._det_buf = None
        self._learn_noise = learn_noise
        self._learn_templates = learn_templates
        self._learn_templates_pval = learn_templates_pval

    ## properties

    def get_det(self):
        if self._det is None:
            self._det = self._det_cls(tf=self.tf, *self._det_params[0],
                **self._det_params[1])
            self._det_buf = MxRingBuffer(capacity=self._det_limit,
                dimension=(self.tf * self.nc),
                dtype=self.dtype)
            if self.verbose.has_print:
                print 'build detector:', self._det_cls, self._det_params
        return self._det

    det = property(get_det)

    ## FilterBanksortingNode interface

    def _pre_filter(self):
        pass

    def _post_sort(self):
        self.det.reset()
        self.det(self._chunk)
        spks = self.det.get_extracted_events(
            mc=False, align_kind='min', align_at=self._learn_templates)
        spks_div = self.component_divergence(spks, loading=False,
            with_noise=True)
        spks_new = sp.any(
            sp.stats.chisqprob(spks_div, self.tf * self.nc) <
            self._learn_templates_pval, axis=1)
        if self.verbose.has_print:
            print 'S:', len(spks), 'Snew:', spks_new.sum()
        keep_spks = spks[spks_new]
        if len(keep_spks) > 0:
            self._det_buf.extend(keep_spks)

    def _execute(self, x):
        # call super to get sorting
        rval = super(ABOTMNode, self)._execute(x)
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
        for k in self.get_filter_idx():
            filt = self._bank.bank[k]
            # 1) snr drop below 0.5
            if filt.get_snr < 0.5:
                filt.active = False

            # 2) rate drop below 1.0
            if hasattr(filt, 'rate'):
                try:
                    nspks = len(self.rval[k])
                except:
                    nspks = 0
                nsmpl = self._data.shape[0]
                filt.rate.observation(nspks, nsmpl)
                if filt.rate.estimate() < 1.0:
                    filt.active = False

    def _adapt_filter_new(self):
        if self._det_buf.is_full:
            if self.verbose.has_print:
                print 'det_buf is full!'

            # get all spikes and clear buffer
            spks = self._det_buf[:].copy()
            self._det_buf.clear()

            # processing chain
            flow = (PrewhiteningNode2(self.ce) + PCANode(output_dim=10) +
                    HomoscedasticClusteringNode(
                        clus_type='gmm', debug=self.verbose.has_print))
            flow(spks)
            lbls = flow[-1].labels
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
                    self.create_filter(spks_i.mean(0))
                    if self.verbose.has_print:
                        print 'accepted, with %d spikes' % len(spks_i)
            del flow, spks
        else:
            if self.verbose.has_print:
                print 'det_buf:', self._det_buf

    def _adapt_noise(self):
        if self._learn_noise:
            nep = None
            if self._learn_noise == 'sort':
                if len(self.rval) > 0:
                    nep = epochs_from_spiketrain_set(
                        self.rval, cut=self.tf,
                        end=self._data.shape[0])['noise']
            elif self._learn_noise == 'det':
                if len(self.det.events) > 0:
                    nep = self.det.get_epochs(merge=True, invert=True)
            self.ce.update(self._data, epochs=nep)

    def _adapt_filter_current(self):
        """adapt templates/filters using non overlapping spikes"""

        # checks and inits
        if self._data is None or self.rval is None:
            return
        ovlp_info = overlaps(self.rval, self.tf)[0]
        cut = get_cut(self.tf)

        # adapt filters with found waveforms
        for u in self.rval:
            st = self.rval[u][ovlp_info[u] == False]
            if len(st) == 0:
                continue
            spks_u = get_aligned_spikes(
                self._data, st, cut, self._learn_templates, mc=True,
                kind='min')[0]
            if spks_u.size == 0:
                continue
            self._bank.bank[u].extend_xi_buf(spks_u)

##---MAIN

if __name__ == '__main__':
    pass
