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


"""covariance estimator for timeseries data"""
__docformat__ = 'restructuredtext'
__all__ = ['BaseTimeSeriesCovarianceEstimator', 'TimeSeriesCovE',
           'XcorrStore', 'build_idx_set', 'build_block_toeplitz_from_xcorrs']

##--- IMPORTS

from .util import *
import scipy as sp
from scipy import linalg as sp_la
from .funcs_general import xcorr
from .matrix_loading import (compute_coloured_loading,
                             compute_diagonal_loading,
                             compute_matrix_cond)
from .util import INDEX_DTYPE

##--- CLASSES

class BaseTimeSeriesCovarianceEstimator(object):
    """covariance estimator base class"""

    ## constructor

    def __init__(self,
                 weight=0.05,
                 cond=50,
                 dtype=None, ):
        """
        :Parameters:
            weight : float
                float from [0.0, 1.0]. new observations will be weighted and
                contribute to the update with the factor weight
                Default=0.05
            cond : float
                Condition number to assert if requesting the loaded matrix.
                Default=50
            dtype : numpy.dtype
                anything that can be used as a numpy.dtype object
                Default=None
        """

        # members
        self.dtype = sp.dtype(dtype or sp.float32)

        # privates
        self._weight = weight
        self._cond = float(cond)
        self._is_initialised = False

    ## getter and setter methods

    def get_cmx(self, **kwargs):
        if self._is_initialised is False:
            raise RuntimeError('Estimator has not been initialised!')
        return self._get_cmx(**kwargs)

    def _get_cmx(self, **kwargs):
        raise NotImplementedError

    def get_icmx(self, **kwargs):
        if self._is_initialised is False:
            raise RuntimeError('Estimator has not been initialised!')
        return self._get_icmx(**kwargs)

    def _get_icmx(self, **kwargs):
        raise NotImplementedError

    def get_svd(self, **kwargs):
        if self._is_initialised is False:
            raise RuntimeError('Estimator has not been initialised!')
        return self._get_svd(**kwargs)

    def _get_svd(self, **kwargs):
        raise NotImplementedError

    def get_whitening_op(self, **kwargs):
        if self._is_initialised is False:
            raise RuntimeError('Estimator has not been initialised!')
        return self._get_whitening_op(**kwargs)

    def _get_whitening_op(self, **kwargs):
        raise NotImplementedError

    def get_cond(self, **kwargs):
        if not self.is_initialised:
            raise RuntimeError('Estimator has not been initialised!')
        sv = self._get_svd(**kwargs)[1]
        return compute_matrix_cond(sv)

    def is_cond_ok(self, **kwargs):
        cond = self.get_cond(**kwargs)
        return cond <= self._cond

    def get_cmx_loaded(self, **kwargs):
        if not self.is_initialised:
            raise RuntimeError('Estimator has not been initialised!')
        kind = kwargs.get('kind', 'diagonal')
        if kind not in ['diagonal', 'coloured']:
            raise ValueError(
                'kind must be one of \'diagonal\' or \'coloured\'!')
        cmx = self._get_cmx(**kwargs)
        svd = self._get_svd(**kwargs)
        return {
            'coloured':compute_coloured_loading,
            'diagonal':compute_diagonal_loading,
            }[kind](cmx, svd, self._cond)

    def get_icmx_loaded(self, **kwargs):
        if not self.is_initialised:
            raise RuntimeError('Estimator has not been initialised!')
        lcmx = self.get_cmx_loaded(**kwargs)
        return sp_la.inv(lcmx)

    def is_initialised(self):
        return self._is_initialised

    ## public methods

    def reset(self):
        """reset the internal buffers to None"""

        self._reset()

    def update(self, data, **kwargs):
        """update covariance matrix with epochs of x

        :Parameters:
            data : list or ndarray
                the data to operate on with observations in the rows
            kwargs : dict
        :Exception ValueError:
                if x is not a ndarray of rank 2
        """

        # checks
        data = sp.asarray(data, dtype=self.dtype)
        if data.ndim != 2:
            raise ValueError('data is not of rank 2')

        # relay
        rval = self._update(data, **kwargs)
        if self._is_initialised is False:
            self._is_initialised = bool(rval)

    ## private methods

    def _update(self, data, **kwargs):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    ## special methods

    def __str__(self):
        return '%s(init:%s)' % (self.__class__.__name__, self._is_initialised)


class TimeSeriesCovE(BaseTimeSeriesCovarianceEstimator):
    """covariance estimator for timeseries data

    Given strips of multichanneled data, this covariance estimator is able to
    produce the time-lagged block-covariance-matrix
    Estimates are built by taking auto- and cross-correlations of the
    multichanneled data pieces, with a fixed lag. From this data a toeplitz
    matrix is build for each combination of channels. The estimate is the
    toeplitz block matrix of sub-matrices derived like s.a.
    """

    ## constructor

    def __init__(self,
                 # TimeSeriesCovE
                 tf_max=100,
                 nc=4,
                 # BaseTimeSeriesCovarianceEstimator
                 weight=0.05,
                 cond=50,
                 dtype=None,
                 ):
        """
        :Parameters:
            see BaseTimeSeriesCovarianceEstimator

            tf_max : int
                the maximum lag for the cross-correlation function to
                calculate
                and store internally. the estimator will be able to provide
                covariance matrices for lags in [1..tf_max].
                Default=100
            nc : int
                channel count of expected data. data that is feed to update
                the
                estimator is checked for this channel count. also determines
                 the
                size of the internal storage for the correlation functions.
                Default=4
        """

        # checks
        if tf_max <= 0:
            raise ValueError('tf_max <= 0! has to be > 0')
        if nc <= 0:
            raise ValueError('nc <= 0! has to be > 0')

        # super
        super(TimeSeriesCovE, self).__init__(
            weight=weight,
            cond=cond,
            dtype=dtype)

        # members
        self._tf_max = int(tf_max)
        self._nc = int(nc)
        self._store = XcorrStore(self._tf_max, self._nc)
        self._buf_cmx = {}
        self._buf_icmx = {}
        self._buf_svd = {}
        self._buf_whi = {}
        self._chan_set = []

    ## getter and setter - base

    def _get_cmx(self, **kwargs):
        # keywords:
        #    chan_set : tuple
        #    tf : int

        tf = int(kwargs.get('tf'))
        chan_set = tuple(sorted(kwargs.get('chan_set')))
        if chan_set not in self._chan_set:
            raise ValueError('unknown chan_set: %s' % str(chan_set))
        dtype = kwargs.get('dtype', None)
        buf_key = (tf, chan_set)

        if buf_key not in self._buf_cmx:
            self._buf_cmx[buf_key] = build_block_toeplitz_from_xcorrs(
                tf, chan_set, self._store, dtype=dtype)
        return self._buf_cmx[buf_key]

    def _get_icmx(self, **kwargs):
        # keywords:
        #    chan_set : tuple
        #    tf : int

        tf = int(kwargs.get('tf'))
        chan_set = tuple(sorted(kwargs.get('chan_set')))
        if chan_set not in self._chan_set:
            raise ValueError('unknown chan_set: %s' % str(chan_set))
        buf_key = (tf, chan_set)

        if buf_key not in self._buf_icmx:
            svd = self._get_svd(**kwargs)
            self._buf_icmx[buf_key] = sp.dot(
                sp.dot(svd[0], sp.diag(1. / svd[1])), svd[2])
        return self._buf_icmx[buf_key]

    def _get_svd(self, **kwargs):
        # keywords:
        #    chan_set : tuple
        #    tf : int

        tf = int(kwargs.get('tf'))
        chan_set = tuple(sorted(kwargs.get('chan_set')))
        if chan_set not in self._chan_set:
            raise ValueError('unknown chan_set: %s' % str(chan_set))
        buf_key = (tf, chan_set)

        if buf_key not in self._buf_svd:
            cmx = self._get_cmx(**kwargs)
            self._buf_svd[buf_key] = sp_la.svd(cmx)
        return self._buf_svd[buf_key]

    def _get_whitening_op(self, **kwargs):
        # keywords:
        #    chan_set : tuple
        #    tf : int

        tf = int(kwargs.get('tf'))
        chan_set = tuple(sorted(kwargs.get('chan_set')))
        if chan_set not in self._chan_set:
            raise ValueError('unknown chan_set: %s' % str(chan_set))
        buf_key = (tf, chan_set)

        if buf_key not in self._buf_whi:
            svd = self._get_svd(**kwargs)
            self._buf_whi[buf_key] = sp.dot(
                sp.dot(svd[0], sp.diag(sp.sqrt(1. / svd[1]))), svd[2])
        return self._buf_whi[buf_key]

    # getter and setter - own

    def get_tf_max(self):
        return self._tf_max

    def set_tf_max(self, value):
        if value < 1:
            raise ValueError('tf_max must be >= 1')
        self._tf_max = int(value)
        self.reset()
        # TODO: reset tf_max for self._store

    def get_nc(self):
        return self._nc

    def get_chan_set(self):
        return self._chan_set

    def new_chan_set(self, cs_new):
        if max(cs_new) >= self._nc:
            raise ValueError(
                'new channel set is incompatible with channel count!')
        if cs_new in self._chan_set:
            raise ValueError('channel set already included!')
        self._chan_set.append(tuple(sorted(cs_new)))

    def rm_chan_set(self, cs_rm):
        if tuple(cs_rm) in self._chan_set:
            self._chan_set.remove(tuple(cs_rm))
        else:
            raise ValueError('channel set not included!')

    ## implementation base

    def _update(self, data, **kwargs):
        """updates the estimator with new data

        :Parameters:
            data : ndarray
                the data to operate on
        :Keywords:
            epochs : ndarray
                epochs to delimit the data
        """

        # check data
        if data.shape[1] != self._nc:
            raise ValueError('channel count (columns) must be %d' % self._nc)
        if data.shape[0] < self._tf_max:
            raise ValueError(
                'must give at least %d samples of data' % self._tf_max)

        # check epochs
        epochs = kwargs.get('epochs', None)
        if epochs is None:
            epochs = sp.array([[0, data.shape[0]]], dtype=INDEX_DTYPE)
        else:
            epochs = sp.asarray(epochs)
        len_epoch = epochs[:, 1] - epochs[:, 0]
        if epochs[len_epoch > self._tf_max, :].size == 0:
            raise ValueError('epoch sum too small for update!')
        if epochs[len_epoch > self._tf_max, :].sum() < self._tf_max * 2:
            raise ValueError('epoch fragmentation too high for update')
        epochs = epochs[len_epoch > self._tf_max]
        n_epoch = epochs.shape[0]
        len_epoch = epochs[:, 1] - epochs[:, 0]
        len_epoch_all = len_epoch.sum()
        self._clear_buf()

        # calculate cross-correlation functions for new observation
        processed = {}
        for cs in self._chan_set:
            chan_set = build_idx_set(cs)
            for m, n in chan_set:
                if (m, n) not in processed:
                    processed[m, n] = []
                else:
                    continue
                for e in xrange(n_epoch):
                    chunk = data[epochs[e, 0]:epochs[e, 1]]
                    xc = xcorr(chunk[:, m], chunk[:, n], lag=self._tf_max - 1)
                    processed[m, n].append((len_epoch[e], xc))
        for k in processed.keys():
            processed[k] = sp.vstack(
                [item[0] * item[1] for item in processed[k]]
            ).sum(axis=0) / len_epoch_all
            if k in self._store:
                self._store[k] *= 1.0 - self._weight
                self._store[k] += self._weight * processed[k]
            else:
                self._store[k] = processed[k]

        # return
        return True

    def _clear_buf(self):
        self._buf_cmx.clear()
        self._buf_icmx.clear()
        self._buf_svd.clear()
        self._buf_whi.clear()

    def _reset(self):
        self._store.reset()
        self._clear_buf()
        self._chan_set = []

    @staticmethod
    def std_white_noise_init(tf_max, nc):
        chan_set = tuple(range(nc))
        rval = TimeSeriesCovE(tf_max=tf_max, nc=nc)
        rval.new_chan_set(chan_set)
        for m, n in build_idx_set(chan_set):
            xc = sp.zeros(2 * tf_max - 1)
            if m == n:
                xc[tf_max - 1] = 1
            rval._store[m, n] = xc
        rval._is_initialised = True
        return rval


class XcorrStore(object):
    """storage for cross-correlation functionals"""

    def __init__(self, tf=100, nc=4):
        """
        :Parameters:
            tf : int
                the length of the channel xcorrs
                Default=100
            nc : int
                the channel count for the storage
                Default=4
        """

        # checks
        if tf <= 0:
            raise ValueError('need tf > 1')
        if nc <= 0:
            raise ValueError('need nc > 1')

        # members
        self._tf = int(tf)
        self._nc = int(nc)
        self._store = {}

    def __getitem__(self, key):
        self._check_index(key)
        return self._store[key]

    def __setitem__(self, key, value):
        self._check_index(key)
        self._check_value(value)
        self._store[key] = value

    def __contains__(self, key):
        return key in self._store

    def __iter__(self):
        return self._store.__iter__()

    def _check_index(self, key):
        if not isinstance(key, tuple):
            raise IndexError('needs 2dim index!')
        if len(key) != 2:
            raise IndexError('needs 2dim index')
        if key[1] < key[0]:
            raise KeyError('x-index must be >= y-index!')
        if 0 > key[0] >= self._nc or 0 > key[1] >= self._nc:
            raise IndexError('index out of bounds! nc = %d' % self._nc)

    def _check_value(self, value):
        if not isinstance(value, sp.ndarray):
            raise TypeError('value needs to be ndarray')
        if value.ndim != 1:
            raise ValueError('value needs to be ndim==1')
        if value.size != (self._tf * 2) - 1:
            raise ValueError(
                'value needs to be size==%d' % int(self._tf * 2 - 1))

    def reset(self):
        self._store.clear()


def build_idx_set(ids):
    """builds the block index set for an upper triangular matrix

    :Parameters:
        ids : iterable
    """

    return [(ids[i], ids[j])
    for i in xrange(len(ids))
    for j in xrange(i, len(ids))]


def build_block_toeplitz_from_xcorrs(tf, chan_set, xcorrs, dtype=None):
    """builds a block toeplitz matrix from a set of channel xcorrs

    :Parameters:
        tf : int
            desired lag in samples
        chan_set : list
            list of channel ids to build the channel set from. the block
            covarinace matrix will be build so that blocks are ordered from
            lower to higher channel id.
        xcorrs : XcorrStore
            XcorrStore object holding the xcorrs for various channel
            combinations
        dtype : dtype derivable
            this will be passed to the constructor for the matrix returned.
            Default=None
    """

    # inits and checks
    assert tf <= xcorrs._tf
    chan_set = sorted(chan_set)
    nc = len(chan_set)
    assert all(sp.diff(chan_set) >= 1)
    assert max(chan_set) <= xcorrs._nc
    assert all([key in xcorrs for key in
                build_idx_set(chan_set)]), 'no data for requested channels'
    rval = sp.empty((tf * nc, tf * nc), dtype=dtype)

    # build blocks and insert into rval
    for i in xrange(nc):
        m = chan_set[i]
        for j in xrange(i, nc):
            n = chan_set[j]
            xc = xcorrs[m, n]
            sample0 = xc.size / 2
            r = xc[sample0:sample0 + tf]
            c = xc[sample0 + 1 - tf:sample0 + 1][::-1]
            block_ij = sp_la.toeplitz(c, r)
            rval[i * tf:(i + 1) * tf, j * tf:(j + 1) * tf] = block_ij
            if i != j:
                rval[j * tf:(j + 1) * tf, i * tf:(i + 1) * tf] = block_ij.T

    # return
    return rval

##--- MAIN

if __name__ == '__main__':
    dlen = 10000
    tf_max = 67
    nc = 4

    my_data = [sp.randn(dlen, nc) * (sp.arange(4) + 1),
               sp.randn(dlen, nc) * (sp.arange(4) + 5),
               sp.randn(dlen, nc) * (sp.arange(4) + 9)]

    E = TimeSeriesCovE(tf_max=tf_max, nc=4)
    E.new_chan_set((0, 1, 2, 3))
    E.new_chan_set((1, 2))
    E.update(my_data[0])
    E.update(my_data[1])
    E.update(my_data[1], epochs=[[0, 100], [1000, 5000], [9500, 9745]])
    print E

    Calltf67_params = {'tf':67, 'chan_set':(0, 1, 2, 3)}
    Calltf67 = E.get_cmx(**Calltf67_params)
    print Calltf67
    print Calltf67.shape
    print E.get_svd(**Calltf67_params)
    print E.get_cond(**Calltf67_params)

    C12tf67_params = {'tf':20, 'chan_set':(1, 2)}
    C12tf67 = E.get_cmx(**C12tf67_params)
    print C12tf67
    print C12tf67.shape
    print E.get_svd(**C12tf67_params)
    print E.get_cond(**C12tf67_params)

    iC12tf67 = E.get_cmx(**C12tf67_params)
    print iC12tf67
    print iC12tf67.shape

    whiC12tf67 = E.get_whitening_op(**C12tf67_params)
    print whiC12tf67
    print whiC12tf67.shape

    from spikeplot import plt

    plt.matshow(Calltf67)
    plt.colorbar(ticks=range(16))
    plt.matshow(C12tf67)
    plt.colorbar(ticks=range(16))
    plt.matshow(iC12tf67)
    plt.colorbar(ticks=range(16))
    plt.matshow(whiC12tf67)
    plt.colorbar(ticks=range(16))
    plt.show()
