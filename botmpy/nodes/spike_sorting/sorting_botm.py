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

"""spike sorting using the BOTM algorithm"""
__docformat__ = "restructuredtext"
__all__ = ["BayesOptimalTemplateMatchingNode", "BOTMNode"]

## IMPORTS

import logging
import scipy as sp
from scipy import linalg as sp_la
from sklearn.mixture import log_multivariate_normal_density
from sklearn.utils.extmath import logsumexp
from ...common import shifted_matrix_sub, mcvec_to_conc, epochs_from_binvec, merge_epochs, matrix_argmax, get_cut
from .base_sorting import FilterBankSortingNode

## CLASSES

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

        :param kwargs:
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
        :type spk_pr_bias: tuple
        :keyword spk_pr_bias: will only be used when the resolution method is
            'sic'. After a spike has been found in a spike epoch, its
            discriminant will be biased by the `bias` for `extend` samples.

            Default=None
        """

        # kwargs
        ovlp_taus = kwargs.pop('ovlp_taus', None)
        noi_pr = kwargs.pop("noi_pr", 1e0)
        spk_pr = kwargs.pop("spk_pr", 1e-6)
        spk_pr_bias = kwargs.pop("spk_pr_bias", None)

        # super
        super(BayesOptimalTemplateMatchingNode, self).__init__(**kwargs)

        # members
        self._ovlp_taus = ovlp_taus
        if self._ovlp_taus is not None:
            self._ovlp_taus = list(self._ovlp_taus)
            if self.verbose.has_print:
                print "using overlap channels"
        else:
            if self.verbose.has_print:
                print "using subtractive interference cancelation"
        self._disc = None
        self._pr_n = None
        self._lpr_n = None
        self._pr_s = None
        self._lpr_s = None
        self._pr_s_b = None
        self._oc_idx = None

        self.noise_prior = noi_pr
        self.spike_prior = spk_pr
        self.spike_prior_bias = spk_pr_bias

    ## properties

    def get_noise_prior(self):
        return self._pr_n

    def set_noise_prior(self, value):
        if value <= 0.0:
            raise ValueError("noise prior <= 0.0")
        if value > 1.0:
            raise ValueError("noise prior > 1.0")
        self._pr_n = float(value)
        self._lpr_n = sp.log(self._pr_n)

    noise_prior = property(get_noise_prior, set_noise_prior)

    def get_spike_prior(self):
        return self._pr_s

    def set_spike_prior(self, value):
        if value <= 0.0:
            raise ValueError("spike prior <= 0.0")
        if value > 1.0:
            raise ValueError("spike prior > 1.0")
        self._pr_s = float(value)
        self._lpr_s = sp.log(self._pr_s)

    spike_prior = property(get_spike_prior, set_spike_prior)

    def get_spike_prior_bias(self):
        return self._pr_s_b

    def set_spike_prior_bias(self, value):
        if value is None:
            return
        if len(value) != 2:
            raise ValueError("expecting tuple of length 2")
        value = float(value[0]), int(value[1])
        if value[1] < 1:
            raise ValueError("extend cannot be non-positive")
        self._pr_s_b = value

    spike_prior_bias = property(get_spike_prior_bias, set_spike_prior_bias)

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
                                .5 * self.get_xc_at(i))

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
                            self.get_xc_at(f0, f1, tau))
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

        # init
        if self.nf == 0:
            return
        spk_ep = epochs_from_binvec(
            sp.nanmax(self._disc, axis=1) > self._lpr_n)
        if spk_ep.size == 0:
            return
        l, r = get_cut(self._tf)
        for i in xrange(spk_ep.shape[0]):
            # FIX: for now we just continue for empty epochs,
            # where do they come from anyways?!
            if spk_ep[i, 1] - spk_ep[i, 0] < 1:
                continue
            mc = self._disc[spk_ep[i, 0]:spk_ep[i, 1], :].argmax(0).argmax()
            s = self._disc[spk_ep[i, 0]:spk_ep[i, 1], mc].argmax() + spk_ep[
                i, 0]
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
            # method: subtractive interference cancellation
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
                        logging.warn(
                            "more spikes than filters found! "
                            "epoch: [%d:%d] %d" % (
                                spk_ep[i][0] + self._chunk_offset,
                                spk_ep[i][1] + self._chunk_offset,
                                niter))
                        if niter > 2 * self.nf:
                            break

                    # find epoch details
                    ep_t = sp.nanargmax(sp.nanmax(ep_disc, axis=1))
                    ep_c = sp.nanargmax(ep_disc[ep_t])

                    # build subtrahend
                    sub = shifted_matrix_sub(
                        sp.zeros_like(ep_disc),
                        self._xc[ep_c, :, :].T,
                        ep_t - self._tf + 1)

                    # apply subtrahend
                    if ep_fout_norm > sp_la.norm(ep_fout + sub):
                        ## DEBUG

                        if self.verbose.get_has_plot(1):
                            try:
                                from spikeplot import xvf_tensor, plt, COLOURS

                                x_range = sp.arange(
                                    spk_ep[i, 0] + self._chunk_offset,
                                    spk_ep[i, 1] + self._chunk_offset)
                                f = plt.figure()
                                f.suptitle("spike epoch [%d:%d] #%d" %
                                           (spk_ep[i, 0] + self._chunk_offset,
                                            spk_ep[i, 1] + self._chunk_offset,
                                            niter))
                                ax1 = f.add_subplot(211)
                                ax1.set_color_cycle(
                                    ['k'] + COLOURS[:self.nf] * 2)
                                ax1.plot(x_range, sp.zeros_like(x_range),
                                         ls='--')
                                ax1.plot(x_range, ep_disc, label="pre_sub")
                                ax1.axvline(x_range[ep_t], c='k')
                                ax2 = f.add_subplot(212, sharex=ax1, sharey=ax1)
                                ax2.set_color_cycle(['k'] + COLOURS[:self.nf])
                                ax2.plot(x_range, sp.zeros_like(x_range),
                                         ls='--')
                                ax2.plot(x_range, sub)
                                ax2.axvline(x_range[ep_t], c='k')
                            except:
                                pass

                        ## BUGED

                        ep_disc += sub + self._lpr_s
                        if self._pr_s_b is not None:
                            bias, extend = self._pr_s_b
                            ep_disc[ep_t:min(ep_t + extend,
                                             ep_disc.shape[0]), ep_c] -= bias

                        ## DEBUG

                        if self.verbose.get_has_plot(1):
                            try:
                                ax1.plot(x_range, ep_disc, ls=':', lw=2,
                                         label="post_sub")
                                ax1.legend(loc=2)
                            except:
                                pass

                        ## BUGED

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
            raise ValueError("no observations passed!")
        data = []
        if obs.ndim == 2:
            if obs.shape[1] != self._tf * self._nc:
                raise ValueError("data dimensions not compatible with model")
            for i in xrange(obs.shape[0]):
                data.append(obs[i])
        elif obs.ndim == 3:
            if obs.shape[1:] != (self._tf, self._nc):
                raise ValueError("data dimensions not compatible with model")
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
        lpr = log_multivariate_normal_density(data, comps, sigma,
                                              'tied') + prior
        logprob = logsumexp(lpr, axis=1)
        return sp.exp(lpr - logprob[:, sp.newaxis])

    def component_divergence(self, obs, with_noise=False, loading=False,
                             subdim=None):
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
            raise ValueError("no observations passed!")
        data = []
        if obs.ndim == 2:
            if obs.shape[1] != self._tf * self._nc:
                raise ValueError("data dimensions not compatible with model")
            for i in xrange(obs.shape[0]):
                data.append(obs[i])
        elif obs.ndim == 3:
            if obs.shape[1:] != (self._tf, self._nc):
                raise ValueError("data dimensions not compatible with model")
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

## MAIN

if __name__ == "__main__":
    pass

## EOF
