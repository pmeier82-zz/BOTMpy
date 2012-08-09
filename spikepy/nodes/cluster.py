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

"""initialization - clustering of spikes in feature space"""

__docformat__ = 'restructuredtext'
__all__ = ['ClusteringNode', 'HomoscedasticClusteringNode']

##---IMPORTS

import scipy as sp
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from .base_nodes import ResetNode

##---CLASSES

class ClusteringNode(ResetNode):
    """interface for clustering algorithms"""

    ## constructor

    def __init__(self, dtype=sp.float32):
        """
        :type dtype: dtype resolvable
        :param dtype: will be passed to :py:class:`mdp.Node`
        """

        # super
        super(ClusteringNode, self).__init__(dtype=dtype)

        # members
        self.labels = None
        self._labels = None
        self.parameters = None
        self._parameters = None

    ## mdp.Node stuff

    def is_invertable(self):
        return True

    def is_trainable(self):
        return False

    def _reset(self):
        self.labels = None
        self._labels = None
        self.parameters = None
        self._parameters = None


class HomoscedasticClusteringNode(ClusteringNode):
    """clustering with model order selection to learn a mixture model

    Assuming the data are prewhitened spikes, possibly in some condensed
    representation e.g. PCA, the problem is to find the correct number of
    components and their corresponding means. The covariance matrix of all
    components is assumed to be the same, as the variation in the data is
    produced by an additive noise process. Further the covariance matrix can
    be assumed be the identity matrix (or a scaled version due to estimation
    errors, thus a spherical covariance),

    To increase performance, it is assumed all necessary pre-processing has
    been taken care of already, to assure an optimal clustering performance
    (e.g.: alignment, resampling, (noise)whitening, etc.)

    So we have to find the number of components and their means in a
    homoscedastic clustering problem. The 'goodness of fit' will be evaluated
    by evaluating a likelihood based criterion that is penalised for an
    increasing number of model parameters (to prevent overfitting) (ref: BIC).
    Minimising said criterion will lead to the most likely model.
    """

    def __init__(self, clus_type='kmeans', crange=range(1, 16), maxiter=32,
                 repeats=4, conv_th=1e-4, sigma_factor=4.0, dtype=sp.float32,
                 debug=False):
        """
        :type clus_type: str
        :param cluls_type: clustering algorithm to use. Must be one of:
            'kmeans', 'gmm'
            Default='kmeans'
        :type crange: list
        :param crange: cluster count to test for
            Default=range(1,16)
        :type maxiter: int
        :param maxiter: upper bound for the iterations per run
            Default=32
        :type repeats: int
        :param repeats: repeat this many times per cluster count
            Default=4
        :type conv_th: float
        :param conv_th: convergence threshold.
            Default=1e-4
        :type sigma_factor: float
        :param sigma_factor: variance factor for the spherical covariance
            Default=4.0
        :type dtype: dtype resolvable
        :param dtype: dtype for internal calculations
            Default=scipy.float32
        :type debug: bool
        :param debug: if True, announce progress to stdout.
        """

        # super
        super(HomoscedasticClusteringNode, self).__init__(dtype=dtype)

        # members
        self.clus_type = str(clus_type)
        if self.clus_type not in ['kmeans', 'gmm']:
            raise ValueError(
                'clus_type must be one of: \'kmeans\' or \'gmm\'!')
        self.crange = list(crange)
        self.maxiter = int(maxiter)
        self.repeats = int(repeats)
        self.conv_th = float(conv_th)
        self.sigma_factor = float(sigma_factor)
        self._ll = None
        self._gof = None
        self._winner = None
        self.debug = bool(debug)

    def _reset(self):
        super(HomoscedasticClusteringNode, self)._reset()
        self._gof = None
        self._winner = None

    def _execute(self, x, *args, **kwargs):
        """run the clustering on a set of observations"""

        # inits
        self._labels = sp.zeros((len(self.crange) * self.repeats,
                                 x.shape[0]), dtype=sp.integer) - 1
        self._gof = sp.zeros(len(self.crange) * self.repeats,
            dtype=self.dtype)
        self._ll = sp.zeros(len(self.crange) * self.repeats,
            dtype=self.dtype)
        self._parameters = [None] * len(self.crange) * self.repeats

        # clustering
        for c in xrange(len(self.crange)):
            k = self.crange[c]
            for r in xrange(self.repeats):
                # inits
                if self.debug is True:
                    print '\t[c:%d][r:%d]' % (self.crange[c], r + 1),
                idx = c * self.repeats + r

                # evaluate model for this run
                if self.clus_type == 'kmeans':
                    model = KMeans(k=k, init='k-means++',
                        max_iter=self.maxiter)
                    model.fit(x)
                    self._labels[idx] = model.labels_
                    self._parameters[idx] = model.cluster_centers_
                    self._ll[idx] = model.score(x)
                    del model
                if self.clus_type == 'gmm':
                    model = GMM(n_components=k, cvtype='spherical')
                    #model = GMM(n_states=k, cvtype='spherical')
                    model.n_features = self.input_dim
                    model.covars = sp.ones(k) * self.sigma_factor
                    model.fit(x, n_iter=0, init_params='wm')
                    model.fit(x,
                        n_iter=self.maxiter,
                        thresh=self.conv_th,
                        init_params='',
                        params='wm')
                    self._labels[idx] = model.predict(x)
                    self._parameters[idx] = model.means
                    self._ll[idx] = model.score(x).sum()
                    del model

                # evaluate goodness of fit for this run
                self._gof[idx] = self.gof(x, self._ll[idx], k)

                # debug
                if self.debug is True:
                    print self._gof[idx]

        self._winner = sp.nanargmin(self._gof)
        self.parameters = self._parameters[self._winner]
        self.labels = self._labels[self._winner]

    def gof(self, obs, LL, k):
        """evaluate the goodness of fit given the data and labels

        :type obs: ndarray
        :param obs: the observations
        :type LL: float
        :param LL: model log likelihood
        :type k: int
        :param k: number of mixture components
        """

        # components
        N, Nk = map(sp.float64, obs.shape)
        Np = k * (Nk + 1) - 1

        #=============================================================
        # BIC value (Xu & Wunsch, 2005)
        # BIC(K) = LL - (Np / 2) * log(N)
        # chose: arg(K) max BIC(K) = arg(K) min -BIC(K)
        #=============================================================
        return -LL + Np * 0.5 * sp.log(N)

        #=============================================================
        # AIC value (Xu & Wunsch, 2005)
        # AIC(K) = -2 * (N - 1 - Nk - k * 0.5) * LL / N + 3 * Np
        #=============================================================
        # return -2.0 * (N - 1 - Nk - k * .5) * LL / float(N) + 3.0 * Np

    def plot(self, data, views=2, show=False):
        """plot clustering"""

        views = min(views, int(data.shape[1] / 2))

        # get plotting tools
        from spikeplot import plt, cluster

        fig = plt.figure()
        ax = [fig.add_subplot(2, views, v + 1) for v in xrange(views)]
        axg = fig.add_subplot(212)
        ncmp = int(self.labels.max() + 1)
        cdata = dict(zip(xrange(ncmp),
            [data[self.labels == c] for c in xrange(ncmp)]))

        # plot clustering
        for v in xrange(views):
            cluster(cdata,
                data_dim=(2 * v, 2 * v + 1),
                plot_handle=ax[v],
                plot_mean=sp.sqrt(self.sigma_factor),
                xlabel='PC %d' % int(2 * v),
                ylabel='PC %d' % int(2 * v + 1),
                show=False)

        # plot gof
        axg.plot(self._gof, ls='steps')
        for i in xrange(1, len(self.crange)):
            axg.axvline(i * self.repeats - 0.5, c='y', ls='--')
        axg.axvspan(self._winner - 0.5, self._winner + 0.5, fc='gray',
            alpha=0.2)
        labels = []
        for k in self.crange:
            labels += ['%d' % k]
            labels += [''] * (self.repeats - 1)
        axg.set_xticks(sp.arange(len(labels)))
        axg.set_xticklabels(labels)
        axg.set_xlabel('repeats')
        axg.set_ylabel('BIC')
        axg.set_xlim(-1, len(labels))

        # show?
        if show is True:
            plt.show()

##--- MAIN

if __name__ == '__main__':
    pass
