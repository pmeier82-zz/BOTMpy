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


"""python module managing the template/filter set for one recording group

Recording groups correspond to one (potentially multi-channeled) recording
device/site. According to the generative data model [1] we store a set of
prototypical neuronal waveforms (templates, mean waveforms) and a
multi-channeled time-lagged noise covariance matrix. Given that we can derive
a set of corresponding matched filters, to find the template in the noisy
data stream.

Additionally we formulate a feature space, spanned by the first n principal
components of a noise-whitened training data set. Data in this context refers
to cut strips X of fixed length tx from the (multichannel,
nc=channel count) signal data. These cut strips can be interpreted as
vectors in R^(tx*nc) (cf. time series embedding [2]). Cut strips X are first
whitened for the noise correlations, by applying the noise-whiteneing
operator W (C = W^-1' W^-1), leading to Y. The events of the same unit in Y
should follow an approximate multivariate standard normal Gaussian
distribution. Next we project the Y into the first n principal components of
learned from the training set, leading to Z. In Z we can formulate a
Gaussian mixture model to evaluate the data, determine cluster count,
validity and membership of new observation projected into the feature space.

The GMM is adapted by new observations found bz the filters, additionally a
spike detection multi-unit is evaluated against the model to find new
clusters over time.

[more detail?]

[1] Franke 2010, online spike sorting with linear filters
[2] Luetkephol 2005, Timeseries Analysis
!check correct reference!!
"""

__docformat__ = 'restructuredtext'
__all__ = ['FilterGroup', 'BotmFilter']

##---IMPORTS
try:
    from sklearn.mixture.gmm import GMM, lmvnpdf, logsum
except ImportError:
    from scikits.learn.mixture import GMM, lmvnpdf, logsum
import scipy as sp
from nodes import (PCANode, MatchedFilterNode, HomoscedasticClusteringNode)
from common import (get_tau_align_min, MxRingBuffer, mcvec_from_conc,
                    mcvec_to_conc, TimeSeriesCovE)

##---CONSTANTS

# seems ok if we are conservative on the (covariance) with 4sigma!
SIGNIFICANT = 0.05

##---CLASSES

class BotmFilter(MatchedFilterNode):
    """filter for managing"""

    def __init__(self, *args, **kwargs):
        super(BotmFilter, self).__init__(*args, **kwargs)
        self.last_spike_time = -1

    def __str__(self):
        return ''.join([super(BotmFilter, self).__str__(),
                        '[rb=%d, srn=%f]' % (len(self._xi_buf), self.snr)])


class FilterGroup(object):
    """filter group manager class"""

    def __init__(self, gid, params, pca_dim=8, verbose=False):
        """
        :parameters:
            idx : int
                index of filter group
            params : dict
                parameters dicts
            pca_dim : int
                output components for the pca projection
        """

        self.gid = int(gid)
        self.params = params
        self.unit = {}
        self.pca_dim = pca_dim
        self.pca = PCANode(output_dim=self.pca_dim)
        self.whi = None
        self.model = None
        self.model_unit_map = []
        self.ce = TimeSeriesCovE(tf_max=self.params['tf'],
                                 nc=self.params['nc'])
        self.ce.new_chan_set(self.params['cs'])
        self.spk_buf = MxRingBuffer(
            capacity=self.params['train_amount'],
            dimension=self.params['tf'] * self.params['nc'])
        self.verbose = bool(verbose)
        self._ukey = 0
        self._query_buf = None
        self._last_time = -1
        self._spikes_rejected = 0

        self.update = self._train

    # special methods

    def __str__(self):
        status = {True:'TRAINING', False:'SORTING', }[self.is_training]
        rval = [status, 'spks:%d' % len(self.spk_buf), str(self.model)]
        for u in self.unit:
            rval.append('%d :: %s' % (u, str(self.unit[u])))
        print
        print '\n'.join(rval)
        print
        return '\n'.join(rval)

    # properties

    def get_is_training(self):
        return self.model is None

    is_training = property(get_is_training)

    def get_need_promote(self):
        return self.model is not None and self._query_buf is None

    need_promote = property(get_need_promote)

    # update methods

    def _train(self, spks, uids, times):
        """fill multi-unit (uid=0) spike buffer with any incoming spikes

        :type spks: ndarray
        :param spks: spike buffer [nspks,tf,nc]
        :type uids: ndarray
        :param uids: list of unit ids for the spikes in spks
        :type times: ndarray
        :param times: list of time values for the spikes in spks
        """

        if spks.shape[0] > 0:
            spks, sidx = self.aligned_embedding(spks)
            if sidx.any():
                spks = spks[sidx]
                uids = uids[sidx]
                times = times[sidx]
                try:
                    self.spk_buf.extend(spks)
                except:
                    pass
                self._last_time = times.max()
            self._spikes_rejected += sp.sum(sidx == False)
        if self.spk_buf.is_full():
            self._end_training()

    def _end_training(self):
        """end training by estimating clustering model and building filters"""

        if self.ce.is_initialised() is False:
            if self.verbose is True:
                print 'no covariance initialised'
            return
        if self.verbose is True:
            print 'end training'

        # cluster spikes for templates and build filters
        spk_buf_data = self.spk_buf[:]
        cspks = self.feature_space(spk_buf_data)
        clus = HomoscedasticClusteringNode(clus_type='gmm')
        clus(cspks)
        nclus = int(clus.labels.max()) + 1
        if self.verbose is True:
            print 'found', nclus, 'units to begin with'
        for i in xrange(nclus):
            xi = spk_buf_data[clus.labels == i].mean(0)
            self.add_filter(mcvec_from_conc(xi), rebuild=False)

        # start normal operation
        self.rebuild()
        self.spk_buf.clear()
        self.update = self._update

    def _update(self, spks, uids, times):
        """updating internals with spikes

        :type spks: ndarray
        :param spks: spike buffer [nspks,tf,nc]
        :type uids: ndarray
        :param uids: list of unit ids for the spikes in spks
        :type times: ndarray
        :param times: list of time values for the spikes in spks
        """

        if spks.shape[0] > 0:
            if sum(uids == 0) > 0:
                det_spks, sidx = self.aligned_embedding(spks[uids == 0])
                if sidx.any():
                    det_spks = det_spks[sidx]
                    props = self.model.predict_proba(
                        self.feature_space(det_spks))
                    for i in xrange(det_spks.shape[0]):
                        if not (props[i] > SIGNIFICANT).any():
                            self.spk_buf.append(det_spks[i])
            if sum(uids != 0) > 0:
                # need to check for unit id validity
                if sum(uids != 0) > 0:
                    fbg_spks = spks[uids != 0]
                    fbg_times = times[uids != 0]
                    fbg_uids = uids[uids != 0]
                    for i in xrange(fbg_spks.shape[0]):
                        uid = fbg_uids[i]
                        if uid not in self.unit:
                            continue
                        self.unit[uid].append_xi_buf(fbg_spks[i])
                        self.unit[uid].last_spike_time = fbg_times[i]
            self._last_time = times.max()
        if self.spk_buf.is_full():
            self.recluster()

    def update_ce(self, coveblock):
        """update self.ce with COVE block, rebuild all internals"""

        nc, tf, xc = coveblock.data_lst[-4:-1]
        if nc != self.ce.get_nc() or tf != self.ce.get_tf_max():
            raise ValueError('tf and nc does not match')
        self.ce._clear_buf()
        for i in xrange(nc):
            for j in xrange(i, nc):
                self.ce._store[i, j] = xc[i * nc + j]
        if not self.ce.is_initialised():
            self.ce._is_initialised = True
        self.whi = self.ce.get_whitening_op(tf=self.params['tf'],
                                            chan_set=self.params['cs'])
        if self.verbose is True:
            print 'covariance update'
        if self.model is not None:
            self.rebuild()

    # interface methods

    def query(self, mc=True):
        """returns the current set of templates, filters and cov or None"""

        try:
            if self.is_training is False:
                if self._query_buf is None:
                    mod = {
                        True:lambda x:x,
                        False:lambda x:mcvec_to_conc(x)
                    }[mc]
                self._query_buf = (self.params,
                                   self.ce,
                                   self.unit)
                self._spikes_rejected = 0
        finally:
            return self._query_buf

    def add_filter(self, xi, rebuild=True):
        """spawn a new filter with template xi"""

        self._ukey += 1
        self.unit[self._ukey] = BotmFilter(self.params['tf'],
                                           self.params['nc'],
                                           self.ce,
                                           chan_set=self.params['cs'],
                                           rb_cap=self.params['spike_amount'])
        self.unit[self._ukey].fill_xi_buf(xi)
        if rebuild is True:
            self.rebuild()

    ## helpers

    def aligned_embedding(self, spks):
        nspks = spks.shape[0]
        if spks.ndim != 3:
            spks_temp = []
            for i in xrange(nspks):
                spks_temp.append(
                    mcvec_from_conc(
                        spks[i],
                        nc=self.params['nc']))
            spks = sp.asarray(spks_temp)
            del spks_temp
        taus = get_tau_align_min(spks, self.params['ali_at'])
        rval = sp.empty((nspks, self.params['tf'] * self.params['nc']))
        sidx = sp.ones(nspks, bool)
        for i in xrange(nspks):
            try:
                start = -taus[i] - self.params['ali_at']
                end = start + self.params['tf']
                rval[i] = mcvec_to_conc(spks[i, start:end, :])
            except:
                sidx[i] = False
                continue
        return rval, sidx

    def recluster(self):
        """recluster spikes in self.spk_buf"""

        if self.verbose is True:
            print 'starting to recluster'

        if len(self.spk_buf) == 0:
            return
        clus = HomoscedasticClusteringNode(clus_type='gmm', crange=range(1, 4))
        clus(self.feature_space(self.spk_buf[:]))
        nclus = int(clus.labels() + 1)
        for i in xrange(nclus):
            xi = self.spk_buf[clus.labels == i].mean(0)
            # TODO: neue templates evaluieren
            self.add_filter(mcvec_from_conc(xi))
            if self.verbose is True:
                print 'adding unit %d' % self._ukey
        self.spk_buf.clear()
        self.rebuild()

        if self.verbose is True:
            print 'done reclustering'


    def rebuild(self):
        """recalc filters and build gmm model in feature space"""

        # check filters
        for u in self.unit.keys():
            u_snr = self.unit[u].snr
            if self.unit[u].snr < self.params['snr_th']:
                del self.unit[u]
                if self.verbose is True:
                    print 'dropping unit', u, 'with snr', u_snr
                continue
            dt = self._last_time - self.unit[u].last_spike_time
            if dt > self._last_time:
                self.unit[u].last_spike_time = self._last_time
                if self.verbose is True:
                    print 'negative dt for unit %d! reseting to %s' % (
                        u, self._last_time)
                continue
            if ( dt > self.params['spike_timeout']):
                del self.unit[u]
                if self.verbose is True:
                    print 'dropping unit', u, 'with spike timeout'
                continue
            self.unit[u].calc_filter()

        # training again if no units left
        if len(self.unit) == 0:
            self.model = None
            self.model_unit_map = []
            self.spk_buf.clear()
            self.update = self._train
            print 'rebuild done: no units left training again'
            return

        # build model
        n_states = len(self.unit)
        self.model = GMM(n_states=n_states, cvtype='spherical')
        self.model.n_features = self.pca_dim
        self.model.covars = sp.array([self.params['sigma']] * n_states)
        self.model_unit_map = self.unit.keys()
        means = sp.vstack([mcvec_to_conc(self.unit[u].xi)
                           for u in self.model_unit_map])
        self.model.means = self.feature_space(means)
        self._query_buf = None
        if self.verbose is True:
            print 'rebuild done:', self.model

    def feature_space(self, obs):
        return self.pca(sp.dot(obs, self.whi))

    @staticmethod
    def greedy_match(a, b, tau):
        return a in xrange(b - tau, b + tau + 1)

    @staticmethod
    def greedy_simi(st1, st2, tau):
        st1i = 0
        st2i = 0
        rval = 0
        while st1i < st1.size and st2i < s2.size:
            rval += FilterGroup.greedy_match(st1[st1i], st2[st2i], tau)
            if st1[st1i] < st2[st2i]:
                st1i += 1
            else:
                st2i += 1
        return 2.0 * rval / (st1.size + st2.size)

##---MAIN

if __name__ == '__main__':
    pass
