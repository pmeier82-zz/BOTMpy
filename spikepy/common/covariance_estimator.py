# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012 Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Neural Information Processing Group (NI)
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


"""covariance estimator for timeseries data"""
__docformat__ = 'restructuredtext'
__all__ = ['BaseTimeSeriesCovarianceEstimator', 'TimeSeriesCovE',
           'XcorrStore', 'build_idx_set', 'build_block_toeplitz_from_xcorrs']

##--- IMPORTS

import scipy as sp
from scipy import linalg as sp_la
from scipy import random as sp_rd
from collections import deque
from .funcs_general import xcorr
from .matrix_ops import (compute_coloured_loading, compute_diagonal_loading,
                         compute_matrix_cond)
from .util import INDEX_DTYPE

##--- CLASSES

class BaseTimeSeriesCovarianceEstimator(object):
    """covariance estimator base class"""

    ## constructor

    def __init__(self, weight=0.05, cond=50, dtype=None):
        """
        :type weight: float
        :param weight: from [0.0, 1.0]. new observations will be weighted and
            contribute to the update with the factor weight. (exp model)
            Default=0.05
        :type cond: float
        :param cond: condition number to assert if the loaded
            matrix is requested.
            Default=50
        :type dtype: dtype resolvable
        :param dtype: anything that can be used as a numpy.dtype object
            Default=float32
        """

        # members
        self.dtype = sp.dtype(dtype or sp.float32)

        # privates
        self._weight = weight
        self._cond = float(cond)
        self._is_initialised = False
        self._n_upd = 0
        self._n_upd_smpl = 0

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
                   'coloured': compute_coloured_loading,
                   'diagonal': compute_diagonal_loading,
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

        self._is_initialised = False
        self._n_upd = 0
        self._n_upd_smpl = 0
        self._reset()

    def update(self, data, **kwargs):
        """update covariance matrix with epochs of :data:

        :type data: ndarray
        :param data: data vector [samples, channels]
        :exception ValueError: `data` is not a ndarray with `ndim == 2`
        """

        # checks
        data = sp.asarray(data, dtype=self.dtype)
        if data.ndim != 2:
            raise ValueError('data is not of rank 2')

        # relay
        n_smpl = self._update(data, **kwargs)
        if self._is_initialised is False:
            self._is_initialised = n_smpl > 0
        if n_smpl > 0:
            self._n_upd += 1
            self._n_upd_smpl += n_smpl

    ## private methods

    def _update(self, data, **kwargs):
        # should return the number of samples that went into building the new
        # observation
        raise NotImplementedError

    def _reset(self):
        pass

    ## special methods

    def __str__(self, additional=''):
        return '%s(init=%s%s)' % (self.__class__.__name__,
                                  self._is_initialised,
                                  additional)


class TimeSeriesCovE(BaseTimeSeriesCovarianceEstimator):
    """covariance estimator for timeseries data

    Given strips of (multi-channeled) data, this covariance estimator is able
    to estimate the time-lagged (block-)covariance-matrix, which has a
    (block-)toeplitz structure.
    Estimates are built by taking (cross- and) auto-correlations of the
    (multi-channeled) data pieces, for all lags in the defined range. From
    this data a toeplitz matrix is build (for each combination of channels).
    """
    # TODO: glibber glibber - better doc text

    ## constructor

    def __init__(self, tf_max=100, nc=4, weight=0.05, cond=50,
                 with_default_chan_set=True, dtype=None):
        """see BaseTimeSeriesCovarianceEstimator

        :type tf_max: int
        :param tf_max: the maximum lag for the cross-correlation functions
            internally stored. the estimator will be able to provide
            covariance matrices for lags in [1..tf_max].
            Default=100
        :type nc: int
        :param nc: channel count of expected data. data that is feed to update
            the estimator is checked for this channel count. also determines
            the size of the internal storage for the correlation functions.
            Default=4
        :type with_default_chan_set: bool
        :param with_default_chan_set: if True, add the default channel set
        """

        # checks
        if tf_max <= 0:
            raise ValueError('tf_max <= 0')
        if nc <= 0:
            raise ValueError('nc <= 0')

        # super
        super(TimeSeriesCovE, self).__init__(weight=weight, cond=cond,
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

        # init
        if with_default_chan_set is True:
            self.new_chan_set(tuple(range(self._nc)))

    ## getter and setter - base

    def _process_keywords(self, kwargs):
        """return parameters from keywords

        :type kwargs: dict
        :param kwargs: keyword parameter dict
        :rtype: tuple
        :returns: tf, chan_set, dtype
        """

        try:
            tf = int(kwargs.get('tf'))
        except:
            tf = int(self._tf_max)
        try:
            cs = tuple(kwargs.get('chan_set'))
        except:
            cs = tuple(range(self._nc))
        return tf, cs

    def _get_cmx(self, **kwargs):
        """yield the current estimate

        :type chan_set: tuple
        :keyword chan_set: channel ids forming a valid channel set
        :type tf: int
        :keyword tf: max lags in samples
        :returns: ndarray - (block-) toeplitz covariance matrix
        """

        tf, chan_set = self._process_keywords(kwargs)
        buf_key = (tf, chan_set)
        if buf_key not in self._buf_cmx:
            self._buf_cmx[buf_key] = build_block_toeplitz_from_xcorrs(
                tf, chan_set, self._store, dtype=self.dtype)
        return self._buf_cmx[buf_key]

    def _get_icmx(self, **kwargs):
        """yield the inverse of the current estimate

        :type chan_set: tuple
        :keyword chan_set: channel ids forming a valid channel set
        :type tf: int
        :keyword tf: max lags in samples
        :returns: ndarray - inverse (block-) toeplitz covariance matrix
        """

        tf, chan_set = self._process_keywords(kwargs)
        buf_key = (tf, chan_set)
        if buf_key not in self._buf_icmx:
            svd = self._get_svd(**kwargs)
            self._buf_icmx[buf_key] = sp.dot(
                sp.dot(svd[0], sp.diag(1. / svd[1])), svd[2])
        return self._buf_icmx[buf_key]

    def _get_svd(self, **kwargs):
        """yield the singular value decomposition of the current estimate

        :type chan_set: tuple
        :keyword chan_set: channel ids forming a valid channel set
        :type tf: int
        :keyword tf: max lags in samples
        :returns: tuple - U, s, Vh as returned by :scipy.linalg.svd:
        """

        tf, chan_set = self._process_keywords(kwargs)
        buf_key = (tf, chan_set)
        if buf_key not in self._buf_svd:
            cmx = self._get_cmx(**kwargs)
            self._buf_svd[buf_key] = sp_la.svd(cmx)
        return self._buf_svd[buf_key]

    def _get_whitening_op(self, **kwargs):
        """yield the whitening operator with respect to the current
        estimate for observation from the vector space this matrix operates
        on.

        if C = Q.T * Q then Q^-1 is the whitening operator

        calculated via SVD

        :type chan_set: tuple
        :keyword chan_set: channel ids forming a valid channel set
        :type tf: int
        :keyword tf: max lags in samples
        :returns: ndarray - whitening operator matrix
        """

        tf, chan_set = self._process_keywords(kwargs)
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
            raise ValueError('tf_max < 1')
        self._tf_max = int(value)
        self.reset()
        # TODO: reset tf_max for self._store

    tf_max = property(get_tf_max)

    def get_nc(self):
        return self._nc

    nc = property(get_nc)

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

    ## special methods

    def __str__(self):
        return super(TimeSeriesCovE, self).__str__(',tf=%s,nc=%s' %
                                                   (self._tf_max, self._nc))

    ## implementation base

    def _update(self, data, **kwargs):
        """updates the estimator with new data

        :type data: ndarray
        :param data: the data to operate on [samples, channels]
        :type epochs: ndarray
        :keyword epochs: epochs delimiting the data to take the estimate over
        :type min_len: int
        :keyword min_len: minimum length of epochs in samples
        :rtype: int
        :returns: number of samples that went into the sample
        """

        # kwargs
        epochs = kwargs.get('epochs', None)
        min_len = kwargs.get('min_len', 3 * self._tf_max)

        # check data
        if data.shape[1] != self._nc:
            raise ValueError('channel count (columns) must be %d' % self._nc)
        if data.shape[0] < min_len:
            raise ValueError('must give at least %d samples of data' % min_len)

        # check epochs
        if epochs is None:
            epochs = sp.array([[0, data.shape[0]]], dtype=INDEX_DTYPE)
        else:
            epochs = sp.asarray(epochs)
        len_epoch = epochs[:, 1] - epochs[:, 0]
        if not any(len_epoch >= min_len):
            raise ValueError('no epochs with len >= min_len!')
        epochs = epochs[len_epoch > min_len]
        n_epoch = epochs.shape[0]
        len_epoch = epochs[:, 1] - epochs[:, 0]
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
                    xc = xcorr(data[epochs[e, 0]:epochs[e, 1], m],
                        data[epochs[e, 0]:epochs[e, 1], n],
                        lag=self._tf_max - 1,
                        normalise=True)
                    processed[m, n].append(xc * len_epoch[e])
        for k in processed.keys():
            processed[k] = sp.sum(processed[k], axis=0) / len_epoch.sum()
            if k in self._store:
                self._store[k] *= 1.0 - self._weight
                self._store[k] += self._weight * processed[k]
            else:
                self._store[k] = processed[k]

        # return
        return len_epoch.sum()

    def _clear_buf(self):
        self._buf_cmx.clear()
        self._buf_icmx.clear()
        self._buf_svd.clear()
        self._buf_whi.clear()

    def _reset(self):
        self._store.reset()
        self._clear_buf()
        # self._chan_set = [] # setting to default chan_set
        self._chan_set = [tuple(range(self._nc))]

    @staticmethod
    def white_noise_init(tf_max, nc, std=1.0, dtype=None):
        chan_set = tuple(range(nc))
        rval = TimeSeriesCovE(tf_max=tf_max, nc=nc)
        for m, n in build_idx_set(chan_set):
            xc = sp.zeros(2 * tf_max - 1, dtype=dtype or sp.float32)
            if m == n:
                xc[tf_max - 1] = float(std * std)
            rval._store[m, n] = xc
        rval._is_initialised = True
        return rval


class TimeSeriesCovE2(TimeSeriesCovE):
    """additional representations of the underlying xcorr container"""

    def __init__(self, *args, **kwargs):
        # super
        super(TimeSeriesCovE2, self).__init__(*args, **kwargs)

        # members
        self._sample_vars = None

    def get_cov_ten(self, **kwargs):
        tf, chan_set = self._process_keywords(kwargs)
        return build_cov_tensor_from_xcorrs(tf, chan_set, self._store, dtype=self.dtype, both=kwargs.get('both', True))


    def _clear_buf(self):
        super(TimeSeriesCovE2, self)._clear_buf()
        self._sample_vars = None
        self._sample_mem = None
        self._sample_coef = None

    def sample(self, n=1):
        """sample with ar model corresponding to the current estimate"""

        # need to fit?
        if self._sample_vars is None:
            print 'fitting VAR model..',
            #self._sample_vars = LWR(self.get_cov_ten(tf=min(12, self._tf_max), both=False))
            self._sample_vars = LWR(self.get_cov_ten(both=False))
            self._sample_coef = sp.hstack([self._sample_vars[0][..., i] for i in xrange(self._sample_vars[0].shape[2])])
            mem_size = self._sample_vars[0].shape[2] * self._nc
            self._sample_mem = deque([0] * self._tf_max * self._nc, maxlen=mem_size)
            self._sample(5000)
            print 'done!'
        return self._sample(n)

    def _sample(self, n=1):
        """sampling"""

        rval = sp_rd.multivariate_normal(sp.zeros(self._nc), self._sample_vars[1], n)
        for k in xrange(self._sample_vars[0].shape[2]):
            rval[k] += sp.dot(self._sample_coef, self._sample_mem)
            self._sample_mem.extendleft(rval[k, ::-1])
        return rval


class XcorrStore(object):
    """storage for cross-correlations"""

    def __init__(self, tf=100, nc=4):
        """
        :type tf: int
        :param tf: length of the channel xcorrs in samples
            Default=100
        :type nc: int
        :param nc: channel count for the storage
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

    :type ids: iterable
    :param ids: set of channel ids
    """

    return [(ids[i], ids[j])
            for i in xrange(len(ids))
            for j in xrange(i, len(ids))]


def build_block_toeplitz_from_xcorrs(tf, chan_set, xcorrs, dtype=None):
    """builds a block toeplitz matrix from a set of channel xcorrs

    :type tf: int
    :param tf: desired lag in samples
    :type chan_set: list
    :param chan_set: list of channel ids to build the channel set from. the
        block covariance matrix will be build so that blocks are ordered from
        lower to higher channel id.
    :type xcorrs: XcorrStore
    :param xcorrs: XcorrStore object holding the xcorrs for various channel
        combinations
    :type dtype: dtype derivable
    :param dtype: will be passed to the constructor for the matrix returned.
        Default=None
    """

    # init and checks
    assert tf <= xcorrs._tf
    chan_set = sorted(chan_set)
    nc = len(chan_set)
    assert all(sp.diff(chan_set) >= 1)
    assert max(chan_set) < xcorrs._nc
    assert all([key in xcorrs for key in
                build_idx_set(chan_set)]), 'no data for requested channels'
    rval = sp.empty((tf * nc, tf * nc), dtype=dtype)

    # build blocks and insert into fout
    for i in xrange(nc):
        m = chan_set[i]
        for j in xrange(i, nc):
            n = chan_set[j]
            xc = xcorrs[m, n]
            sample0 = xc.size / 2
            r = xc[sample0:sample0 + tf]
            c = xc[sample0 + 1 - tf:sample0 + 1][::-1]
            #c = xc[sample0:sample0 - tf:-1]
            block_ij = sp_la.toeplitz(c, r)
            rval[i * tf:(i + 1) * tf, j * tf:(j + 1) * tf] = block_ij
            if i != j:
                rval[j * tf:(j + 1) * tf, i * tf:(i + 1) * tf] = block_ij.T

    # return
    return rval


def build_cov_tensor_from_xcorrs(tf, chan_set, xcorrs, dtype=None, both=False):
    """builds a covariance tensor from a set of channel xcorrs

    The tensor will hold the forward (positive lags) covariances for all auto-/cross-correlations in the chan_set

    :type tf: int
    :param tf: desired lag in samples
    :type chan_set: list
    :param chan_set: list of channel ids to build the channel set from. the
        covvariance tensor will be build so that the chan_set is indexed natively.
    :type xcorrs: XcorrStore
    :param xcorrs: XcorrStore object holding the xcorrs for various channel
        combinations
    :type dtype: dtype derivable
    :param dtype: will be passed to the constructor for the matrix returned.
        Default=None
    """

    # init and checks
    assert tf <= xcorrs._tf
    chan_set = sorted(chan_set)
    nc = len(chan_set)
    assert all(sp.diff(chan_set) >= 1)
    assert max(chan_set) < xcorrs._nc
    assert all([key in xcorrs for key in
                build_idx_set(chan_set)]), 'no data for requested channels'
    xc_len = tf + both * (tf - 1)
    rval = sp.empty((nc, nc, xc_len), dtype=dtype)

    # write single xcorrs
    for i in xrange(nc):
        m = chan_set[i]
        for j in xrange(i, nc):
            n = chan_set[j]
            xc = xcorrs[m, n]
            sample0 = xc.size / 2
            bakw = xc[:sample0 + 1][:tf - 1:-1]
            comb = None
            if both is True:
                rval[i, j, :] = xc[sample0 - tf + 1:sample0 + tf]
            else:
                rval[i, j, :] = xc[sample0:][:tf]
            if i != j:
                if both is True:
                    rval[j, i, :] = xc[::-1][sample0 - tf + 1:sample0 + tf]
                else:
                    rval[j, i, :] = xc[::-1][:tf]

    # return
    return rval


def LWR(R):
    """multivariate Levinson-Durbin recursion, (Whittle, Wiggins and Robinson) aka LWR

    We assume all quantities to be well behaving: real, pos_sem_def, yadda yadda

    Reference:
    "Covariance characterization by partial autocorrelation matrices",
    Morf, Vieira and Kailath, The Annals or Statistics 1978, Vol. 6, No. 3, 643-648

    :type R: ndarray
    :param R: xcorr sequence (nc, nc, N+1), should cover one more lag than the desired model order
    :rtype: ndarray, ndarray
    :return: ar coefficient sequence (nc, nc, N); driving process covariance estimate (nc, nc)
    """

    # R is (nc, nc, N+1)
    N = R.shape[2] - 1
    nc = R.shape[0]
    # coefficients
    A = sp.zeros((nc, nc, N)) # forward (ar)
    B = sp.zeros((nc, nc, N)) # backward (lp)
    # coefficient error covariances
    err_e = sp.zeros((nc, nc)) # forward prediction error covariance
    err_e = R[..., 0]
    err_r = sp.zeros((nc, nc)) # backward prediction error covariance
    err_r = R[..., 0]
    # intermediate update term
    Delta = sp.empty((nc, nc))

    # iterate
    for n in xrange(N):
        # (9)
        # \Delta_{n+1} = R_{n+1} + \sum_{i=1}^{n} A_{n,i} R_{n+1-i}
        Delta[:] = R[..., n + 1]
        for i in xrange(1, n + 1):
            Delta += sp.dot(A[..., i - 1], R[..., n + 1 - i])

        # intermediate for (7), (8), (10), (11)
        # \Delta_{n+1} (\sigma_{n}^{r})^{-1}
        delta_err_r_inv = sp_la.solve(err_r, Delta)
        #delta_err_r_inv = sp.dot(Delta, la.inv(err_r))
        # \Delta_{n+1}^{T} (\sigma_{n}^{\epsilon})^{-1}
        deltaT_err_e_inv = sp_la.solve(err_e, Delta.T)
        #deltaT_err_e_inv = sp.dot(Delta.T, la.inv(err_e))

        AA = A.copy()
        # (7)
        # A_{n+1} = [I,A_{n,1},..,A_{n,n},0] - \Delta_{n+1}(\sigma_{n}^{r})^{-1} [0,B_{n,n},..,B_{n,1},I]
        for i in xrange(1, n + 1):
            A[..., i - 1] -= sp.dot(delta_err_r_inv, B[..., n - i])
        A[..., n] = -delta_err_r_inv

        # (8)
        # B_{n+1} = [0,B_{n,n},..,B_{n,1},I] - \Delta_{n+1}^{T}(\sigma_{n}^{\epsilon})^{-1} [I,A_{n,1},..,A_{n,n},0]
        for i in xrange(1, n + 1):
            B[..., i - 1] -= sp.dot(deltaT_err_e_inv, AA[..., n - i])
        B[..., n] = -deltaT_err_e_inv

        # (10)
        # \sigma_{n+1}^{\epsilon} = \sigma_{n}^{\epsilon} - \Delta_{n+1}(\sigma_{n}^{r})\Delta_{n+1}^{T}
        err_e -= sp.dot(delta_err_r_inv, Delta.T)
        # (11)
        # \sigma_{n+1}^{r} = \sigma_{n}^{r} - \Delta_{n+1}^{T}(\sigma_{n}^{\epsilon})\Delta_{n+1}
        err_r -= sp.dot(deltaT_err_e_inv, Delta)

    # return
    return A, err_e

##--- MAIN

if __name__ == '__main__':
    pass
