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

import sklearn.cluster
import scipy as sp
import scipy.linalg as sp_la
from tables import openFile
from mdp.nodes import PCANode
import cPickle

try:
    import spikeplot as plot
    plot.plt.interactive(False)
    WITH_PLOT = True
except ImportError:
    WITH_PLOT = False


ARC_PATH = './data.h5'

# def get_data(tf=65, trials=5, snr=0.5, mean_correct=False, save=False):
#     # inits
#     db = MunkSession()
#     id_ana_mu = 1284
#     # get relevant info for analysis item
#     q = db.query("""
#     SELECT a.expid, a.tetrode
#     FROM analysis a
#     WHERE a.id = %d
#     """ % id_ana_mu) # L014 tet7 mua
#     id_exp = q[0][0]
#     id_tet = q[0][1]
#     trial_ids = db.get_trial_range_exp(id_exp, trlidx=(0, trials),
#                                        include_error_trials=False)
#     id_mu = db.get_units_for_analysis(id_ana_mu)[0]
#     ndet = TimeSeriesCovE(tf_max=tf, nc=4)
#     data = {}
#     align_at = int(tf / 4)
#
#     print 'loading data trials..'
#     for id_trl in trial_ids:
#         trial_data = None
#         try:
#             trial_data = db.get_tetrode_data(id_trl, id_tet)
#             if mean_correct is True:
#                 trial_data -= trial_data.mean(axis=0)
#             data[id_trl] = trial_data
#             print '\tprocessed %s' % db.get_fname_for_id(id_trl)
#         except Exception, e:
#             raise RuntimeError('error processing %s\n%s' %
#                                (db.get_fname_for_id(id_trl), e))
#         finally:
#             del trial_data
#     print 'done.'
#
#     print 'retrieving multiunit spike set @tf=%d' % tf
#     spks_info = []
#     spks = []
#     for id_trl in trial_ids:
#         trial_st = None
#         try:
#             trial_st = db.get_unit_data(id_mu, id_trl)['spiketrain']
#             if trial_st.size == 0:
#                 print '\tno spiketrain for %s' % db.get_fname_for_id(id_trl)
#                 continue
#             trial_spks, trial_st = get_aligned_spikes(
#                 data[id_trl],
#                 trial_st,
#                 tf,
#                 align_at=align_at,
#                 mc=False,
#                 kind='min')
#             end = data[id_trl].shape[0]
#             nep = epochs_from_spiketrain(trial_st, tf, end=end)
#             nep = invert_epochs(nep, end=end)
#             nep = merge_epochs(nep)
#             ndet.update(data[id_trl], epochs=nep)
#             spks.append(trial_spks)
#             spks_info.append(sp.vstack([[id_trl] * trial_st.size,
#                                         trial_st]).T)
#             print '\tprocessed %s' % db.get_fname_for_id(id_trl)
#         except Exception, e:
#             raise RuntimeError('error processing %s\n%s' %
#                                (db.get_fname_for_id(id_trl), e))
#         finally:
#             del trial_st
#     spks_info = sp.vstack(spks_info)
#     spks = sp.vstack(spks)
#     print 'found %d spikes in total' % spks.shape[0]
#     print 'done.'
#
#     print 'checking SNR of spikes'
#     spks_snr = snr_maha(spks, ndet.get_icmx(tf=tf))
#     good_spks = spks_snr > snr
#     n_spks = spks.shape[0]
#     spks = spks[good_spks]
#     spks_info = spks_info[good_spks].astype(int)
#     print 'keeping %d of %d spikes with SNR > %f' % (spks.shape[0], n_spks,
#                                                      snr)
#
#     if save is True:
#         ndet_pkl = cPickle.dumps(ndet)
#         arc = openFile(ARC_PATH, 'w')
#         arc.createArray(arc.root, 'spks', spks)
#         arc.createArray(arc.root, 'spks_info', spks_info)
#         arc.createArray(arc.root, 'ndet_pkl', ndet_pkl)
#         arc.close()
#     return spks, spks_info, ndet


def load_data():
    arc = openFile(ARC_PATH, 'r')
    spks = arc.getNode('/spks').read()
    spks_info = arc.getNode('/spks_info').read()
    ndet_pkl = arc.getNode('/ndet_pkl').read()
    arc.close()
    ndet = cPickle.loads(ndet_pkl)
    print 'loaded', spks.shape[0], 'spikes from hdf archive'
    return spks, spks_info, ndet


def pre_processing(spks, ndet, tf, pca_dim=4):
    print 'starting to prewhiten w.r.t. noise..',
    spks_prw = sp.dot(spks, ndet.get_whitening_op(tf=tf))
    print 'done.'

    print 'pca projection: %s' % pca_dim,
    spks_pca = PCANode(output_dim=pca_dim)(spks_prw)
    print 'done.'

    return spks_pca


def cluster_kmeans(obs, crange=range(1, 21)):
    rval = None
    winner = sp.inf
    print 'starting multiple runs of: kmeans'
    for i in crange:
        clus = sklearn.cluster.KMeans(k=i)
        clus.fit(obs)
        x = gof(clus.score(obs), obs, i)
        print i, x
        if x < winner:
            winner = x
            rval = clus.labels_
    print 'done.'
    return rval


def cluster_gmm(obs, crange=range(1, 21)):
    rval = None
    winner = sp.inf
    print 'starting multiple runs of: GMM'
    for i in crange:
        clus = sklearn.mixture.GMM(n_components=i, cvtype='spherical')
        clus.fit(obs, n_iter=0, init_params='wm', params='')
        clus.covars = [4.0] * i
        clus.fit(obs, init_params='', params='wm')
        x = gof(clus.score(obs).sum(), obs, i)
        print i, x
        if x < winner:
            winner = x
            rval = clus.predict(obs)
    print 'done.'
    return rval


def cluster_ward(obs, crange=range(1, 21)):
    rval = None
    winner = sp.inf
    print 'starting multiple runs of: hirachial clustering'
    for i in crange:
        clus = sklearn.cluster.Ward(n_clusters=i)
        clus.fit(obs, n_iter=0, init_params='wm', params='')
        clus.covars = [4.0] * i
        clus.fit(obs, init_params='', params='wm')
        x = gof(clus.score(obs).sum(), obs, i)
        print i, x
        if x < winner:
            winner = x
            rval = clus.predict(obs)
    print 'done.'
    return rval


def gof(ll, data, k):
    N, Nk = map(sp.float64, data.shape)
    Np = k * (Nk + 1) - 1

    #=============================================================
    # # calculate BIC value (Xu & Wunsch, 2005)
    # return - ll + Np * 0.5 * sp.log(N)
    #=============================================================

    #=============================================================
    # # calculate AIC value (Xu & Wunsch, 2005)
    # return - 2 * (N - 1 - Nk - ncmp * 0.5) * ll / N + 3 * Np
    #=============================================================

    return - ll + Np * 0.5 * sp.log(N)


def gaussian_heat_kernel(X, delta=1.0):
    return sp.exp(- X ** 2 / (2. * delta ** 2))


def cluster_spectral(obs):
    print 'starting spectral clustering with gaussian heat kernel'
    nobs = obs.shape[0]
    aff_mx = sp.zeros((nobs, nobs))
    for i in xrange(nobs):
        for j in xrange(i, nobs):
            aff_mx[i, j] = sp_la.norm(obs[i] - obs[j])
            if i != j:
                aff_mx[j, i] = aff_mx[i, j]
    simi = gaussian_heat_kernel(aff_mx)

    clus = sklearn.cluster.SpectralClustering()
    clus.fit(simi)
    print 'done.'


def main():
    TF, SNR, PCADIM = 65, 0.5, 8
    NTRL = 10
    LOAD = False
    if LOAD is True:
        spks, spks_info, ndet = load_data()
    else:
        # spks, spks_info, ndet = get_data(tf=TF, trials=NTRL, snr=SNR,
        #                                  mean_correct=False, save=True)
        pass

    # plot.waveforms(spks, tf=TF, show=False)

    input_obs = pre_processing(spks, ndet, TF, pca_dim=PCADIM)
    plot.cluster(input_obs, show=False)

    # kmeans
    labels_km = cluster_kmeans(input_obs)
    obs_km = {}
    wf_km = {}
    for i in xrange(labels_km.max() + 1):
        obs_km[i] = input_obs[labels_km == i]
        wf_km[i] = spks[labels_km == i]
    if WITH_PLOT:
        plot.cluster(obs_km, title='kmeans', show=False)
        plot.waveforms(obs_km, tf=TF, title='kmeans', show=False)

    # gmm
    labels_gmm = cluster_gmm(input_obs)
    obs_gmm = {}
    wf_gmm = {}
    for i in xrange(labels_km.max() + 1):
        obs_gmm[i] = input_obs[labels_gmm == i]
        wf_gmm[i] = spks[labels_gmm == i]
    if WITH_PLOT:
        plot.cluster(obs_gmm, title='gmm', show=False)
        plot.waveforms(wf_gmm, tf=TF, title='gmm', show=False)

    # ward
    labels_ward = cluster_ward(input_obs)
    obs_ward = {}
    wf_ward = {}
    for i in xrange(labels_km.max() + 1):
        obs_ward[i] = input_obs[labels_ward == i]
        wf_ward[i] = spks[labels_ward == i]
    if WITH_PLOT:
        plot.cluster(obs_ward, title='ward', show=False)
        plot.waveforms(wf_ward, tf=TF, title='ward', show=False)

    # spectral
    #cluster_spectral(spks)

    if WITH_PLOT:
        plot.plt.show()

if __name__ == '__main__':
    main()
