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
__all__ = ['ClusteringNode','HomoscedasticClusteringNode']

##---IMPORTS

import scipy as sp
from scikits.learn.mixture import GMM, lmvnpdf, logsum
from scikits.learn.cluster import KMeans
from .base_nodes import ResetNode

##---CLASSES

class ClusteringNode(ResetNode):
    """clustering node interface class"""

    ## constructor

    def __init__(self, dtype=sp.float32):
        """
        :Parameters:
            dtype : scipy.dtype
                dtype for internal calculations
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
        # reset members
        self.labels = None
        self._labels = None
        self.parameters = None
        self._parameters = None

    def _execute(self, x, *args, **kwargs):
        # start clustering
        self._clustering(x, *args, **kwargs)

        # return
        return x

    def _clustering(self, data, **kwargs):
        """abstract clustering method - should be implemented in subclass"""

        raise NotImplementedError

#class GMMwithGOF(GMM):
#    """GMM implementing a bayesian information criterion"""
#
##    def bic(self, data):
##        """calculate the models goodness of fit given data
##
##        This evaluates the Bayesian Information Criterion (BIC) of Schwarz
# for
##        the data, given the current model parameters.
##
##        IC := -n * LL(data) +
##
##        :Parameters:
##            data : ndarray
##                observation, one obs per row
##        """
##
##        return 5 * self.n_states * sp.log(data.shape[0]) - \
##            data.shape[0] * self.score(data).sum()
#
#    def gof(self, data, mode):
#        """calculate goodness of fit"""
#
#        if not hasattr(self, str(mode)):
#            raise ValueError('no such gof-criterion: %s' % str(mode))
#        return getattr(self, str(mode))(data)
#
#    def df(self):
#        """return the degrees of freedom of the model"""
#
#        return {
#            'full' : self.n_states * (self.n_features + 1 + self.n_features
# ** 2 / 2) - 1,
#            'diag' : self.n_states * (self.n_features * 2 + 1) - 1,
#            'spherical' : self.n_states * (self.n_features + 2) - 1,
#            'tied' : self.n_states * (self.n_features + 1) - 1,
#        }[self._cvtype]
#
#    def bic(self, data):
#        """calculate the models goodness of fit given data using the BIC"""
#
#        return self.df() * sp.log(data.shape[0]) - 2 * self.score(data).sum()
#
#    def aic(self, data):
#        """calculate the models goodness of fit given data using the AIC"""
#
#        return self.df() * 2 - 2 * self.score(data).sum()


class HomoscedasticClusteringNode(ClusteringNode):
    """clustering with model oder selection to come up with a component model

    Assuming the data are prewhitened spikes, possibly in some condensed
    representation e.g. PCA, the problem is to find the correct number of
    components and their corresponding means. The covariance matrix of all
    components is assumed to be the same, as the variation in the data is
    produced by the additive noise process. Further the covariance matrix can
    be assumed be the identity matrix (or a scaled version due to estimation
    errors, thus a spherical covariance),

    To increase performance, it is assumed all necessary preprocessing
    measures
    have been taken, to assure an optimal clustering performance (like:
    alignment, resampling, (noise)whitening, etc.)

    So we have to find the number of components and their means in a
    homoscedastic clustering problem. The 'goodness of fit' will be evaluated
    by evaluating a likelihood based criterion that is penalised for an
    increasing number of model parameters (to prevent overfitting) (ref: BIC).
    Minimising said criterion will lead to the most likely model.
    """

    def __init__(self, clus_type='kmeans', crange=range(1, 16), maxiter=32,
                 repeats=4, conv_th=1e-4, sigma_factor=4.0,
                 weights_uniform=False, dtype=sp.float32, debug=False):
        """
        :Parameters:
            clus_type : str
                str giving the clustering algorithm to use. Must be one of:
                'kmeans', 'gmm'
                Default='kmeans'
            crange : list
                cluster count to test for
                Default=range(1,16)
            maxiter : int
                upper bound for the iterations
                Default=32
            repeats : int
                repeat this many times per cluster count
                Default=4
            conv_th : float
                Convergence threshold.
                Default=1e-4
            sigma_factor : float
                variance factor for the spherical covariance
                Default=4.0
            weights_uniform : bool
                If True, use uniform weight when evaluating the goodness of
                fit.
            dtype : scipy.dtype
                dtype for internal calculations
                Default=scipy.float32
            debug : bool
                If True, announce progress to stdout.
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
        self._gof = None
        self._winner = None
        self.debug = bool(debug)
        self.weigths_uniform = bool(weights_uniform)

    def _reset(self):
        self._gof = None
        self._winner = None

        # super
        super(HomoscedasticClusteringNode, self)._reset()

    def _clustering(self, x, *args, **kwargs):
        # inits
        self._labels = sp.zeros(
            (len(self.crange) * self.repeats, x.shape[0]),
            dtype=self.dtype
        ) - 1
        self._gof = sp.zeros(
            len(self.crange) * self.repeats,
            dtype=self.dtype
        )
        self._parameters = [None] * len(self.crange) * self.repeats

        # cluster range
        for c in xrange(len(self.crange)):
            # repeat range
            for r in xrange(self.repeats):
                # debug
                if self.debug is True:
                    print '\t[c:%d][r:%d]' % (self.crange[c], r + 1),

                # parameters for this run
                idx = c * self.repeats + r
                k = self.crange[c]

                # evaluate model for this run
                if self.clus_type == 'kmeans':
                    model = KMeans(k=k, init='k-means++',
                                   max_iter=self.maxiter)
                    model.fit(x)
                    self._labels[idx] = model.labels_
                    self._parameters[idx] = model.cluster_centers_
                    del model
                if self.clus_type == 'gmm':
                    model = GMM(n_states=k, cvtype='spherical')
                    model.n_features = self.input_dim
                    model.covars = sp.ones(model.n_states) * self.sigma_factor
                    model.fit(x, n_iter=0, init_params='wm')
                    params = 'm'
                    if not self.weigths_uniform:
                        params += 'w'
                    model.fit(x,
                              n_iter=self.maxiter,
                              thresh=self.conv_th,
                              init_params='',
                              params=params)
                    self._labels[idx, :] = model.predict(x)
                    self._parameters[idx] = model.means
                    del model

                # evaluate goodness of fit for this run
                self._gof[idx] = self.gof(x,
                                          self._labels[idx, :],
                                          weights_uniform=self
                                          .weigths_uniform)

                # debug
                if self.debug is True:
                    print self._gof[idx]

        self._winner = sp.nanargmin(self._gof)
        self.parameters = self._parameters[self._winner]
        self.labels = self._labels[self._winner, :]

    def gof(self, data, labels, weights_uniform=False):
        """evaluate the goodness of fit given the data and labels

        :Parameters:
            data : ndarray
                The dataset with observations in the rows shape(sp,D)
            labels : ndarray
                The labels for the data shape(sp,)
            weights_uniform : bool
                If true, use uniform weights for the evaluation.
        """

        # inits
        ncmp = int(labels.max() + 1)
        mean = sp.vstack([data[labels == c, :].mean(axis=0)
                          for c in xrange(ncmp)])
        ll = lmvnpdf(data,
                     mean,
                     sp.ones(ncmp) * self.sigma_factor,
                     'spherical')
        w = sp.log(sp.ones(ncmp) / ncmp)
        if not weights_uniform:
            w = sp.array([data[labels == c, :].shape[0] / float(data.shape[0])
                          for c in xrange(ncmp)])

        # components
        ll = logsum(ll + sp.log(w), axis=1).sum()
        N, Nk = map(sp.float64, data.shape)
        Np = ncmp * Nk
        if not weights_uniform:
            Np += ncmp - 1

        #=======================================================================
        # # calculate BIC value (Xu & Wunsch, 2005)
        # return - ll + Np * 0.5 * sp.log(sp)
        #=======================================================================

        #=======================================================================
        # # calculate AIC value (Xu & Wunsch, 2005)
        # return - 2 * (sp - 1 - Nk - ncmp * 0.5) * ll / sp + 3 * Np
        #=======================================================================

        return - ll + Np * 0.5 * sp.log(N)

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
            labels += ['%d-%d' % (k, r + 1) for r in xrange(self.repeats)]
        axg.set_xticks(sp.arange(len(labels)))
        axg.set_xticklabels(labels)
        axg.set_xlim(-1, len(labels))

        # show?
        if show is True:
            plt.show()

##--- MAIN

if __name__ == '__main__':
    mul = 2.0
    dim = 6
    data = sp.vstack(
        [sp.randn(50 * (i + 1), dim) + [5 * i * (-1) ** i] * dim for i in
         xrange(5)]) * mul
    HCN = HomoscedasticClusteringNode(clus_type='gmm',
                                      crange=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                                      maxiter=128,
                                      repeats=4,
                                      sigma_factor=mul * mul,
                                      weights_uniform=False,
                                      debug=True)
    HCN(data)
    HCN.plot(data, views=3, show=True)
