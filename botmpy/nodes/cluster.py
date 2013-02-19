# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012 Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Philipp Meier <pmeier82@gmail.com>
#               Neural Information Processing Group (NI)
#               School for Electrical Engineering and Computer Science
#               Berlin Institute of Technology
#               MAR 5-6, Marchstr. 23, 10587 Berlin, Germany
#               http://www.ni.tu-berlin.de/
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

"""initialization - clustering of spikes in feature space"""

__docformat__ = 'restructuredtext'
__all__ = ['ClusteringNode', 'HomoscedasticClusteringNode']

##---IMPORTS

import scipy as sp
from sklearn.mixture import DPGMM, GMM, VBGMM
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import euclidean_distances
from .base_nodes import ResetNode

##---CLASSES

class FitlerbankParameters(object):
    """collection of initial parameters for a filterbank

    holds templates, the length of templates in samples, number of channels,
    the covariance matrix and the templates itself.
    """

    def __init__(self):
        """

        :return:
        :rtype:
        """

        self.tf = None
        self.nc = None
        self.templates = None
        self.covest = None


class ClusteringNode(ResetNode):
    """interface for clustering algorithms"""

    ## constructor

    def __init__(self, dtype=None):
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

    ## constructor

    def __init__(
            self, clus_type='kmeans', crange=range(1, 16), repeats=4,
            sigma_factor=4.0, max_iter=None, conv_thresh=None, alpha=None,
            cvtype='tied', gof_type='bic', dtype=None, debug=False):
        """
        :type clus_type: str
        :param clus_type: clustering algorithm to use. Must be one of:
            'kmeans', 'gmm'

            Default='kmeans'
        :type crange: list
        :param crange: cluster count to test for

            Default=range(1,16)
        :type repeats: int
        :param repeats: repeat this many times per cluster count

            Default=4
        :type sigma_factor: float
        :param sigma_factor: variance factor for the spherical covariance

            Default=4.0
        :type max_iter: int
        :param max_iter: upper bound for the iterations per run

            Default=None
        :type conv_thresh: float
        :param conv_thresh: convergence threshold.

            Default=None
        :type alpha: float
        :param alpha: alpha value for the variational inference based gmm
            algorithms

            Default=None
        :type cvtype: str
        :param cvtype: covariance type, one of {'spherical', 'diag', 'tied',
            'full'}

            Default='tied'
        :type dtype: dtype resolvable
        :param dtype: dtype for internal calculations

            Default=None
        :type gof_type: str
        :param gof_type: goodness of fit criterion to use, one of {'aic', 'bic'}

            Default='bic'
        :type debug: bool
        :param debug: if True, announce progress to stdout.

            Default=False
        """

        # super
        super(HomoscedasticClusteringNode, self).__init__(dtype=dtype)

        # members
        self._ll = None
        self._gof = None
        self._winner = None
        self.clus_type = str(clus_type)
        self.gof_type = str(gof_type)
        if self.clus_type not in ['kmeans', 'gmm', 'dpgmm']:
            raise ValueError(
                'clus_type must be one of: \'kmeans\', \'gmm\'!')
        self.cvtype = str(cvtype)
        self.crange = list(crange)
        self.repeats = int(repeats)
        self.sigma_factor = float(sigma_factor)
        self.debug = bool(debug)

        self.clus_kwargs = {}
        if max_iter is not None and clus_type in ['kmeans', 'gmm', 'vbgmm',
                                                  'dpgmm']:
            self.clus_kwargs.update(max_iter=max_iter)
        if conv_thresh is not None and clus_type in ['kmeans', 'gmm', 'dpgmm']:
            self.clus_kwargs.update(conv_thresh=conv_thresh)
        if alpha is not None and clus_type in ['vbgmm', 'dpgmm']:
            self.clus_kwargs.update(alpha=alpha)

    def _reset(self):
        super(HomoscedasticClusteringNode, self)._reset()
        self._gof = None
        self._winner = None

    ## spectral clustering

    @staticmethod
    def gauss_heat_kernel(x, delta=1.0):
        return sp.exp(-x ** 2 / (2. * delta ** 2))

    def _fit_spectral(self, x):
        # FIXME: broken still
        D = euclidean_distances(x, x)
        A = HomoscedasticClusteringNode.gauss_heat_kernel(D)
        # clustering
        for c in xrange(len(self.crange)):
            k = self.crange[c]
            for r in xrange(self.repeats):
                # init
                if self.debug is True:
                    print '\t[%s][c:%d][r:%d]' % (
                        self.clus_type, self.crange[c], r + 1),
                idx = c * self.repeats + r

                # evaluate model
                model = SpectralClustering(k=k)
                model.fit(A)
                self._labels[idx] = model.labels_
                means = sp.zeros((k, x.shape[1]))
                for i in xrange(k):
                    means[i] = x[model.labels_ == i].mean(0)
                self._parameters[idx] = means

    ## kmeans

    def _fit_kmeans(self, x):
        # clustering
        for c in xrange(len(self.crange)):
            k = self.crange[c]
            for r in xrange(self.repeats):
                # info
                if self.debug is True:
                    print '\t[%s][c:%d][r:%d]' % (
                        self.clus_type, self.crange[c], r + 1),
                idx = c * self.repeats + r

                # fit kmeans model
                model_kwargs = {}
                if 'max_iter' in self.clus_kwargs:
                    model_kwargs.update(max_iter=self.clus_kwargs['max_iter'])
                model = KMeans(n_clusters=k, init='k-means++', **model_kwargs)
                model.fit(x)
                self._labels[idx] = model.labels_
                self._parameters[idx] = model.cluster_centers_

                # build equivalent gmm
                model_gmm = GMM(n_components=k, covariance_type='spherical')
                model_gmm.means_ = model.cluster_centers_
                model_gmm.covars_ = sp.ones(
                    (k, self.input_dim)) * self.sigma_factor
                model_gmm.weights_ = sp.array(
                    [(self._labels[idx] == i).sum() for i in xrange(k)])

                # evaluate goodness of fit
                self._ll[idx] = model_gmm.score(x).sum()
                if self.gof_type == 'aic':
                    self._gof[idx] = model_gmm.aic(x)
                if self.gof_type == 'bic':
                    self._gof[idx] = model_gmm.bic(x)

                # debug info
                if self.debug is True:
                    print self._gof[idx], model.inertia_

    ## gmm (vanilla em)

    def _fit_gmm(self, x):
        # clustering
        for c in xrange(len(self.crange)):
            k = self.crange[c]
            for r in xrange(self.repeats):
                # info
                if self.debug is True:
                    print '\t[%s][c:%d][r:%d]' % (self.clus_type, k, r + 1),
                idx = c * self.repeats + r

                # fit and evaluate model
                model_kwargs = {}
                if 'conv_thresh' in self.clus_kwargs:
                    model_kwargs.update(thresh=self.clus_kwargs['conv_thresh'])
                if 'max_iter' in self.clus_kwargs:
                    model_kwargs.update(n_iter=self.clus_kwargs['max_iter'])
                model = GMM(
                    n_components=k,
                    covariance_type=self.cvtype,
                    params='wmc',
                    init_params='wmc',
                    **model_kwargs)
                model.covars_ = {'spherical': sp.ones((k, self.input_dim)),
                                 'diag': sp.ones((k, self.input_dim)),
                                 'tied': sp.eye(self.input_dim),
                                 'full': sp.array([sp.eye(self.input_dim)] * k),
                                }[self.cvtype] * self.sigma_factor
                model.fit(x)
                self._labels[idx] = model.predict(x)
                self._parameters[idx] = model.means_
                self._ll[idx] = model.score(x).sum()

                # evaluate goodness of fit
                self._ll[idx] = model.score(x).sum()
                if self.gof_type == 'aic':
                    self._gof[idx] = model.aic(x)
                if self.gof_type == 'bic':
                    self._gof[idx] = model.bic(x)

                # debug
                if self.debug is True:
                    print self._gof[idx], model.converged_

    ## gmm (variational inference bias)
    # FIXME: broken due to sklearn interface change

    def _fit_vbgmm(self, x):
        # clustering
        for c in xrange(len(self.crange)):
            k = self.crange[c]
            for r in xrange(self.repeats):
                # info
                if self.debug is True:
                    print '\t[%s][c:%d][r:%d]' % (
                        self.clus_type, self.crange[c], r + 1),
                idx = c * self.repeats + r

                # fit and evaluate model
                model_kwargs = {}
                if 'alpha' in self.clus_kwargs:
                    model_kwargs.update(alpha=self.clus_kwargs['alpha'])
                if 'conv_thresh' in self.clus_kwargs:
                    model_kwargs.update(thresh=self.clus_kwargs['conv_thresh'])
                model = VBGMM(n_components=k, covariance_type=self.cvtype,
                              **model_kwargs)
                model.n_features = self.input_dim
                fit_kwargs = {}
                if 'max_iter' in self.clus_kwargs:
                    fit_kwargs.update(n_iter=self.clus_kwargs['max_iter'])
                model.fit(x, params='wmc', init_params='wmc', **fit_kwargs)
                self._labels[idx] = model.predict(x)
                self._parameters[idx] = model.means
                self._ll[idx] = model.score(x).sum()

                # evaluate goodness of fit
                self._gof[idx] = self.gof(x, self._ll[idx], k)

                # debug
                if self.debug is True:
                    print self._gof[idx], model.converged_

    ## gmm (Dirichlet process fitting)
    # FIXME: broken due to sklearn interface changes

    def _fit_dpgmm(self, x):
        # clustering
        k = max(self.crange)
        for r in xrange(self.repeats):
            # info
            if self.debug is True:
                print '\t[%s][c:%d][r:%d]' % (self.clus_type, k, r + 1),

            # fit and evaluate model
            model_kwargs = {}
            if 'alpha' in self.clus_kwargs:
                model_kwargs.update(alpha=self.clus_kwargs['alpha'])
            if 'conv_thresh' in self.clus_kwargs:
                model_kwargs.update(thresh=self.clus_kwargs['conv_thresh'])
            if 'max_iter' in self.clus_kwargs:
                model_kwargs.update(n_iter=self.clus_kwargs['max_iter'])

            model = DPGMM(n_components=k, covariance_type=self.cvtype,
                          **model_kwargs)
            model.fit(x)
            self._labels[r] = model.predict(x)
            self._parameters[r] = model.means_
            #self._ll[r] = model.score(x).sum()

            # evaluate goodness of fit for this run
            #self._gof[r] = self.gof(x, self._ll[r], k)
            if self.gof_type == 'aic':
                self._gof[r] = model.aic(x)
            if self.gof_type == 'bic':
                self._gof[r] = model.bic(x)

            # debug
            if self.debug is True:
                print self._gof[r], model.n_components, model.weights_.shape[0]

    ## mdp.node interface

    def _execute(self, x, *args, **kwargs):
        """run the clustering on a set of observations"""

        # init
        self._labels = sp.zeros((len(self.crange) * self.repeats,
                                 x.shape[0]), dtype=int) - 1
        self._gof = sp.zeros(len(self.crange) * self.repeats,
                             dtype=self.dtype)
        self._ll = sp.zeros(len(self.crange) * self.repeats,
                            dtype=self.dtype)
        self._parameters = [None] * len(self.crange) * self.repeats

        # clustering
        fit_func = {'kmeans': self._fit_kmeans,
                    'gmm': self._fit_gmm,
                    #'vbgmm': self._fit_vbgmm,
                    'dpgmm': self._fit_dpgmm,
                    'spectral': self._fit_spectral}[self.clus_type]
        fit_func(x)

        self._winner = sp.nanargmin(self._gof)
        self.parameters = self._parameters[self._winner]
        self.labels = self._labels[self._winner]

    ## plot interface

    def plot(self, data, views=2, show=False):
        """plot clustering"""

        # get plotting tools
        try:
            from spikeplot import plt, cluster
        except ImportError:
            return None

        # init
        views = min(views, int(data.shape[1] / 2))
        fig = plt.figure()
        fig.suptitle('clustering [%s]' % self.clus_type)
        ax = [fig.add_subplot(2, views, v + 1) for v in xrange(views)]
        axg = fig.add_subplot(212)
        ncmp = int(self.labels.max() + 1)
        cdata = dict(zip(xrange(ncmp),
                         [data[self.labels == c] for c in xrange(ncmp)]))

        # plot clustering
        for v in xrange(views):
            cluster(
                cdata,
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
            labels += ['.'] * (self.repeats - 1)
        axg.set_xticks(sp.arange(len(labels)))
        axg.set_xticklabels(labels)
        axg.set_xlabel('cluster count and repeats')
        axg.set_ylabel(str(self.gof_type).upper())
        axg.set_xlim(-1, len(labels))

        # show?
        if show is True:
            plt.show()
        return True

##--- MAIN

if __name__ == '__main__':
    pass
