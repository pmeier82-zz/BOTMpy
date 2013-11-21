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

"""spike sorting using hte BOTM algorithm - adapring over time"""
__docformat__ = "restructuredtext"
__all__ = ["AdaptiveBayesOptimalTemplateMatchingNode", "ABOTMNode"]

## IMPORTS

import collections
import logging
import scipy as sp
from ...common import (
    epochs_from_spiketrain_set, mcvec_to_conc, MxRingBuffer, mcvec_from_conc, get_aligned_spikes, vec2ten,
    get_tau_align_min, get_tau_align_max, get_tau_align_energy, mad_scaling, mad_scale_op_mx, mad_scale_op_vec)
from ..base_node import PCANode
from ..cluster import HomoscedasticClusteringNode
from ..prewhiten import PrewhiteningNode
from ..spike_detection import SDMneoNode
from .sorting_botm import BayesOptimalTemplateMatchingNode

## CONSTANTS

MTEO_DET = SDMneoNode
MTEO_KWARGS = {'k_values': [3, 9, 15, 21],
               'threshold_factor': 0.98,
               'min_dist': 32}

## CLASSES

class AdaptiveBayesOptimalTemplateMatchingNode(BayesOptimalTemplateMatchingNode):
    """Adaptive BOTM Node

    adaptivity here means,backwards sense, that known templates and
    covariances are adapted local temporal changes. In the forward sense a
    parallel spike detection is matched to find currently unidenified units
    in the data.
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
        :keyword det_limit: capacity of the ringbuffer to hold the unexplained
            spikes.

            Default=2000
        :type det_forget: int
        :keyword det_forget: Unexplained spikes that are older than this
            amount of samples will be forgotten. A reclustering to find
            new nodes will be started if ``det_limit`` unexplained spikes
            are found during ``det_forget`` samples. If this value is 0,
            no reclustering will occur.

            Default=1000000
        :type clus_num_reclus: int or list
        :type clus_num_reclus: Number of clusters that will be used in a
            reclustering of unexplained spikes.

            Default: 4
        :type clus_min_size: int
        :keyword clus_min_size: Minimum number of spikes in a cluster of
            unexplained spikes for a new unit to be created from that cluster
            during reclustering.

            Default=50
        :type clus_use_amplitudes: bool
        :keyword clus_use_amplitudes: Determines if amplitudes (max-min) for
            each channel are used in addition to PCA features for clustering.

            Default=True
        :type clus_pca_features: int
        :keyword clus_pca_features: The number of PCA features to use during
            clustering.

            Default=10

        :type clus_algo: str
        :keyword clus_algo: Name of the clustering algorithm to use.
            Allowed are all names HomoscedasticClusteringNode can use,
            e.g. 'gmm' or 'meanshift'.

            Default='gmm'

        :type clus_params: dict
        :keyword clus_params: Dictionary of parameters for chosen algorithm.
            Contents depend on clustering algorithm:

            * 'gmm'
              * 'min_clusters' Minimum number of clusters to try.
                Default=1
              * 'max_clusters' Maximum number of clusters to try.
                Default=14
            * 'mean_shift'
              * Empty.

        :type clus_merge_rsf: int
        :keyword clus_params: Resampling factor used for realignment before
        checking
            if clusters should be merged.

            Default=16

        :type clus_merge_dist: float
        :keyword clus_merge_dist: Maximum euclidean distance between two
        clusters
            that will be merged. Set to 0 to turn off automatic cluster merging.

            Default=0.0

        :type minimum_snr: float
        :keyword minimum_snr: Templates with a signal to noise ratio below this
            value are dropped.

            Default = 0.5

        :type minimum_rate: float
        :keyword minimum_rate: Templates with a firing rate (in Hertz) below
            this value are dropped.

            Default = 0.1

        :type det_kwargs: dict
        :keyword det_kwargs: keywords for the spike detector that will be
            run in parallel on the data.

            Default=MTEO_KWARGS
        """

        # kwargs
        learn_templates = kwargs.pop('learn_templates', -1)
        learn_templates_rsf = kwargs.pop('learn_templates_rsf', 1.0)
        learn_noise = kwargs.pop('learn_noise', None)
        det_cls = kwargs.pop('det_cls')
        if det_cls is None:
            det_cls = MTEO_DET
        det_kwargs = kwargs.pop('det_kwargs')
        if det_kwargs is None:
            det_kwargs = MTEO_KWARGS
        det_limit = kwargs.pop('det_limit', 4000)

        self._forget_samples = kwargs.pop('det_forget', 1000000)
        self._mad_scaling = kwargs.pop('clus_mad_scaling', False)
        self._min_new_cluster_size = kwargs.pop('clus_min_size', 30)
        self._num_reclus = kwargs.pop('clus_num_reclus', 4)
        self._use_amplitudes = kwargs.pop('clus_use_amplitudes', True)
        self._pca_features = kwargs.pop('clus_pca_features', 10)
        self._cluster_algo = kwargs.pop('clus_algo', 'gmm')
        self._cluster_params = kwargs.pop('clus_params', {})
        self._merge_dist = kwargs.pop('clus_merge_dist', 0.0)
        self._merge_rsf = kwargs.pop('clus_merge_rsf', 16)
        self._external_spike_train = None
        self._minimum_snr = kwargs.pop('minimum_snr', 0.5)
        self._minimum_rate = kwargs.pop('minimum_rate', 0.1)

        # check det_cls
        #if not issubclass(det_cls, ThresholdDetectorNode):
        #    raise TypeError(
        #        '\'det_cls\' of type ThresholdDetectorNode is required!')
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
        self._det_kwargs = det_kwargs
        self._det_limit = int(det_limit)
        self._det_buf = None
        self._det_samples = None
        self._learn_noise = learn_noise
        self._learn_templates = learn_templates
        self._learn_templates_rsf = learn_templates_rsf
        self._sample_offset = 0  # Count how often the sorting was executed
        # Number of samples before unexplained spikes are ignored

        # align at (learn_templates)
        if self._learn_templates < 0:
            self._learn_templates = .25
        if isinstance(self._learn_templates, float):
            if 0.0 <= self._learn_templates <= 1.0:
                self._learn_templates *= self.tf
            self._learn_templates = int(self._learn_templates)

        # for initialisation set correct self._cluster method
        self._cluster = self._cluster_init

        self._det_buf = MxRingBuffer(capacity=self._det_limit,
                                     dimension=(self._tf * self._nc),
                                     dtype=self.dtype)
        # Saves (global) samples of unexplained spike events
        self._det_samples = collections.deque(maxlen=self._det_limit)
        # mad scale value
        if self._mad_scaling is False:
            self._mad_scaling = None
        else:
            self._mad_scaling = 0.0

    ## properties

    def get_det(self):
        if self._det is None:
            self._det = self._det_cls(tf=self._tf, **self._det_kwargs)
            if self.verbose.has_print:
                print self._det
        return self._det

    det = property(get_det)

    ## filter bank sorting interface

    def _event_explained(self, ev, padding=15):
        """check event for explanation by the filter bank"""

        # early exit if no discriminants are present
        if not self._disc.size:
            return False

        # cut relevant piece of the discriminants
        data_ep = ev - self._learn_templates, \
                  ev - self._learn_templates + self.tf
        disc_ep = data_ep[0] + self._tf / 2, \
                  data_ep[1] + self._tf / 2
        if self._external_spike_train is not None:
            disc_ep = (disc_ep[0] - self._chunk_offset,
                       disc_ep[1] - self._chunk_offset)
        if self.verbose.has_plot:
            try:
                from spikeplot import mcdata

                ep = data_ep[0] - padding, disc_ep[1] + padding
                mcdata(
                    data=self._chunk[ep[0]:ep[1]],
                    #other=self._disc[at[0]:at[1]], events=evts,
                    other=self._disc[ep[0]:ep[1]],
                    x_offset=ep[0],
                    events={0: [ev], 1: [data_ep[0] + self._tf]},
                    epochs={0: [data_ep], 1: [disc_ep]},
                    title='det@%s(%s) disc@%s' % (
                        ev, self._learn_templates, ev + self._tf),
                    show=True)
            except ImportError:
                pass
                #self.se_cnt += 1

        start = max(0, disc_ep[0] - padding)
        stop = min(self._disc.shape[0], disc_ep[1] + padding)
        return self._disc[start:stop, :].max() >= 0.0

    def _post_sort(self):
        """check the spike sorting against multi unit"""

        if self._external_spike_train is None:
            self.det.reset()
            self.det(self._chunk, ck0=self._chunk_offset,
                     ck1=self._chunk_offset + len(self._chunk))
            if self.det.events is None:
                return
            events = self.det.events
        else:
            events = self._external_spike_train[sp.logical_and(
                self._external_spike_train >= self._chunk_offset,
                self._external_spike_train < self._chunk_offset + len(
                    self._chunk))]

        events_explained = sp.array([self._event_explained(e) for e in events])
        if self.verbose.has_print:
            print 'spks not explained:', (events_explained == False).sum()
        if sp.any(events_explained == False):
            data = self._chunk
            if self._mad_scaling is not None:
                data = 1.0 / self._mad_scaling * self._chunk.copy()
            spks, st = get_aligned_spikes(
                data, events[events_explained == False],
                tf=self._tf, mc=False, kind=self._align_kind,
                align_at=self._learn_templates, rsf=self._learn_templates_rsf)
            self._det_buf.extend(spks)
            self._det_samples.extend(self._sample_offset + st)

        self._disc = None

    def _execute(self, x, ex_st=None):
        if self._mad_scaling is not None:
            alpha = self._ce._weight
            mad_scale = mad_scaling(x)[1]
            if sp.any(self._mad_scaling):
                self._mad_scaling = (1.0 - alpha) * self._mad_scaling
                self._mad_scaling += alpha * mad_scale
            else:
                self._mad_scaling = mad_scale

                # set the external spike train
        self._external_spike_train = ex_st
        # call super to get sorting
        rval = super(AdaptiveBayesOptimalTemplateMatchingNode, self)._execute(x)
        # adaption
        self._adapt_noise()
        self._adapt_filter_drop()
        self._adapt_filter_current()
        self._adapt_filter_new()
        # learn slow noise statistic changes
        self._sample_offset += x.shape[0]  # Increase sample offset
        return rval

    ## adaption methods

    def _adapt_noise(self):
        if self._learn_noise:
            nep = None
            if self._learn_noise == 'sort':
                if len(self.rval) > 0:
                    nep = epochs_from_spiketrain_set(
                        self.rval,
                        cut=(self._learn_templates,
                             self._tf - self._learn_templates),
                        end=self._data.shape[0])['noise']
            elif self._learn_noise == 'det':
                if self._external_spike_train is not None:
                    nep = epochs_from_spiketrain_set(
                        {666: self._external_spike_train},
                        cut=(self._learn_templates,
                             self._tf - self._learn_templates),
                        end=self._data.shape[0])['noise']
                elif len(self.det.events) > 0:
                    nep = self.det.get_epochs(
                        ## this does not have to be the correct cut for the
                        ## detection events! best would be to do an
                        # alignment here!
                        cut=(self._learn_templates,
                             self._tf - self._learn_templates),
                        merge=True, invert=True)
            else:
                raise ValueError('unrecognised value for learn_noise: %s' % str(
                    self._learn_noise))

            try:
                self._ce.update(self._data, epochs=nep)
            except ValueError, e:
                logging.warn(str(e))

    def _adapt_filter_drop(self):
        nsmpl = self._data.shape[0]
        for u in list(self._idx_active_set):
            # 1) snr drop
            if self.bank[u].snr < self._minimum_snr:
                self.filter_deactivate(u)
                logging.warn('deactivating filter %s, snr' % str(u))

            # 2) rate drop
            if hasattr(self.bank[u], 'rate'):
                try:
                    nspks = len(self.rval[u])
                except:
                    nspks = 0
                self.bank[u].rate.observation(nspks, nsmpl)
                if self.bank[u].rate.filled and \
                                self.bank[u].rate.estimate() < self._minimum_rate:
                    self.filter_deactivate(u)
                    logging.warn('deactivating filter %s, rate' % str(u))
        self._update_internals()

    def _adapt_filter_current(self):
        """adapt templates/filters using non overlapping spikes"""

        # check and init
        if self._data is None or self.rval is None:
            return

        # adapt filters with found waveforms
        for u in self.rval:
            spks_u = self.spikes_u(u, mc=True, exclude_overlaps=True,
                                   align_at=self._learn_templates or -1,
                                   align_kind=self._align_kind,
                                   align_rsf=self._learn_templates_rsf)
            if spks_u.size == 0:
                continue
            self.bank[u].extend_xi_buf(spks_u)
            self.bank[u].rate.observation(spks_u.shape[0], self._data.shape[0])
        print [(u, f.rate.estimate()) for (u, f) in self.bank.items()]

    def _adapt_filter_new(self):
        if self._det_buf.is_full and \
                (self._cluster == self._cluster_init or
                     (self._forget_samples > 0 and
                              self._det_samples[0] > self._sample_offset -
                              self._forget_samples)):
            if self.verbose.has_print:
                print 'det_buf is full!'
            self._cluster()
        else:
            if self.verbose.has_print:
                print 'self._det_buf volume:', self._det_buf

    ## something from robert

    def resampled_mean_dist(self, spks1, spks2):
        """ Caclulate distance of resampled means from two sets of spikes
        """
        # resample and realign means to check distance
        means = {}

        means[0] = mcvec_from_conc(spks1.mean(0), nc=self._nc)
        means[1] = mcvec_from_conc(spks2.mean(0), nc=self._nc)

        if self._merge_rsf != 1:
            for u in means.iterkeys():
                means[u] = sp.signal.resample(
                    means[u], self._merge_rsf * means[u].shape[0])

                if self._align_kind == 'min':
                    tau = get_tau_align_min(
                        sp.array([means[u]]),
                        self._learn_templates * self._merge_rsf)
                elif self._align_kind == 'max':
                    tau = get_tau_align_max(
                        sp.array([means[u]]),
                        self._learn_templates * self._merge_rsf)
                elif self._align_kind == 'energy':
                    tau = get_tau_align_energy(
                        sp.array([means[u]]),
                        self._learn_templates * self._merge_rsf)
                else:
                    tau = 0

                # Realignment shouldn't need to be drastic
                max_dist = 2 * self._merge_rsf
                l = means[u].shape[0]
                if abs(tau) > max_dist:
                    logging.warn(('Could not realign %d, distance: %d ' %
                                  (u, tau)))
                    tau = 0
                means[u] = mcvec_to_conc(
                    means[u][max_dist + tau:l - max_dist + tau, :])
        else:
            means[0] = mcvec_to_conc(means[0])
            means[1] = mcvec_to_conc(means[1])

        return sp.spatial.distance.cdist(
            sp.atleast_2d(means[0]), sp.atleast_2d(means[1]), 'euclidean')

    ## cluster methods

    def _cluster_init(self):
        """cluster step for initialisation"""

        # get all spikes and clear buffers
        spks = self._det_buf[:].copy()
        self._det_buf.clear()
        self._det_samples.clear()

        # noise covariance matrix, and scaling due to median average deviation
        C = self._ce.get_cmx(tf=self._tf, chan_set=self._cs)
        if self._mad_scaling is not None:
            C *= mad_scale_op_mx(self._mad_scaling, self._tf)

        # processing chain
        pre_pro = PrewhiteningNode(ncov=C) + \
                  PCANode(output_dim=self._pca_features)
        sigma_factor = 4.0
        min_clusters = self._cluster_params.get('min_clusters', 1)
        max_clusters = self._cluster_params.get('max_clusters', 14)
        rep = 0 if self._cluster_algo == 'meanshift' else 4
        clus = HomoscedasticClusteringNode(
            clus_type=self._cluster_algo,
            cvtype='full',
            debug=self.verbose.has_print,
            sigma_factor=sigma_factor,
            crange=range(min_clusters, max_clusters + 1),
            max_iter=256, repeats=rep)

        # create features
        if self._use_amplitudes:
            n_spikes = spks.shape[0]
            spks_pp = sp.zeros((n_spikes, self._pca_features + self._nc))
            spks_pp[:, :self._pca_features] = pre_pro(spks)

            all = vec2ten(spks, self._nc)
            all_amp = all.max(axis=1) - all.min(axis=1)

            # Scale amplitude features to a level near pca features
            all_amp *= sigma_factor * 5 / all_amp.max()
            spks_pp[:, self._pca_features:] = all_amp
        else:
            spks_pp = pre_pro(spks)

        # cluster
        clus(spks_pp)
        if self.verbose.is_verbose is True:
            clus.plot(spks_pp, show=True)
        lbls = clus.labels

        if self.verbose.has_plot:
            clus.plot(spks_pp, show=True)

        if self._merge_dist > 0.0:
            merged = True
            while merged:
                merged = False
                for i in sp.unique(lbls):
                    spks_i = spks[lbls == i]

                    #for inner in xrange(i):
                    for inner in sp.unique(lbls):
                        if i >= inner:
                            continue
                        spks_inner = spks[lbls == inner]

                        d = self.resampled_mean_dist(spks_i, spks_inner)
                        if self.verbose.has_print:
                            print 'Distance %d-%d: %f' % (i, inner, d)
                        if d <= self._merge_dist:
                            lbls[lbls == i] = inner
                            if self.verbose.has_print:
                                print 'Merged', i, 'and', inner, '-'
                            merged = True
                            break
                    if merged:
                        break
        if self._mad_scaling is not None:
            # if we have scaled the spikes, rescale to original scale
            spks *= mad_scale_op_vec(1.0 / self._mad_scaling, self._tf)
        for i in sp.unique(lbls):
            spks_i = spks[lbls == i]
            if len(spks_i) < self._min_new_cluster_size:
                self._det_buf.extend(spks_i)
                if self.verbose.has_print:
                    print 'Unit %d rejected, only %d spikes' % (i, len(spks_i))
                continue

            spk_i = mcvec_from_conc(spks_i.mean(0), nc=self._nc)
            self.filter_create(spk_i)
            if self.verbose.has_print:
                print 'Unit %d accepted, with %d spikes' % (i, len(spks_i))
        del pre_pro, clus, spks, spks_pp
        self._cluster = self._cluster_base

    def _cluster_base(self):
        """cluster step for normal operation"""

        # get all spikes and clear buffer
        spks = self._det_buf[:].copy()
        self._det_buf.clear()
        self._det_samples.clear()

        # noise covariance matrix, and scaling due to median average deviation
        C = self._ce.get_cmx(tf=self._tf, chan_set=self._cs)
        if self._mad_scaling is not None:
            C *= mad_scale_op_mx(self._mad_scaling, self._tf)

        # processing chain
        pre_pro = PrewhiteningNode(ncov=C) + \
                  PCANode(output_dim=10)
        clus = HomoscedasticClusteringNode(
            clus_type='gmm',
            cvtype='tied',
            debug=self.verbose.has_print,
            sigma_factor=4.0,
            crange=range(1, self._num_reclus + 1),
            max_iter=256)
        spks_pp = pre_pro(spks)
        clus(spks_pp)
        lbls = clus.labels
        if self.verbose.has_plot:
            clus.plot(spks_pp, show=True)
        if self._mad_scaling is not None:
            spks *= mad_scale_op_vec(1.0 / self._mad_scaling, self._tf)
        for i in sp.unique(lbls):
            if self.verbose.has_print:
                print 'checking new unit:',
            spks_i = spks[lbls == i]

            if len(spks_i) < self._min_new_cluster_size:
                self._det_buf.extend(spks_i)
                if self.verbose.has_print:
                    print 'rejected, only %d spikes' % len(spks_i)
            else:
                spk_i = mcvec_from_conc(spks_i.mean(0), nc=self._nc)
                self.filter_create(spk_i)
                if self.verbose.has_print:
                    print 'accepted, with %d spikes' % len(spks_i)
        del pre_pro, clus, spks, spks_pp

    def _update_mad_value(self, mad):
        """update the mad value if `mad_scaling` is True"""


## shortcut
ABOTMNode = AdaptiveBayesOptimalTemplateMatchingNode

## MAIN

if __name__ == '__main__':
    pass
