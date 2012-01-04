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


from spikeplot import waveforms
from os import path as osp
import scipy as sp
from tables import openFile
from common import (epochs_from_spiketrain, extract_spikes, AtfFile, GdfFile,
                    get_cut, TimeSeriesCovE, invert_epochs)
from nodes import PrewhiteningNode, HomoscedasticClusteringNode, AlignmentNode
from mdp.nodes import PCANode
from util import *


##---CONSTANTS

GTPATH = '/home/phil/Dropbox/MunkShare/Methoden Paper/AuswertungAlleDaten/GroundTruth'
TF = 95
CUTOFFSET = 20


##---FUNCTIONS

def initialization(fname_in, do_ovlps=True):
    """detect events from atf file

    :Parameters:
        fname_in : str
            path to an existing .atf file to read from
        do_ovlps : bool
            if true, include overlaps in the events returned, else exclude them
            Default=True
    :Return:
        list like [spikes, covariance matrix, noise-samples]
    """

    print 'starting file', osp.basename(fname_in),

    # inits
    rval = []

    # read in from atf file
    arc_in = AtfFile(fname_in)
    sweep_data = arc_in.get_data(mode='device', item=1)
    sweep_data -= sweep_data.mean(0)
    arc_in.close()
    del arc_in

    # get groundtruth
    exp, block = get_info_for_fname(osp.basename(fname_in))
    arc_gt = openFile(osp.join(GTPATH, '%s%s_intracSpDetResults.h5' % (exp, block)), 'r')
    trial_no_in_block = EXP_DICT[exp][block].index(osp.basename(fname_in))
    gt = arc_gt.getNode('/singleFileResults/%05d/gdf' % int(trial_no_in_block + 1)).read().astype(int)

    # get events
    ev_u0 = gt[1][gt[0] == 1]
    ev_u1 = gt[1][gt[0] == 2]
    if do_ovlps is False:
        ovlp_u0 = arc_gt.getNode('/singleFileResults/%05d/O/00001' %
                                 int(trial_no_in_block + 1)).read().astype(int)
        ovlp_u1 = arc_gt.getNode('/singleFileResults/%05d/O/00002' %
                                 int(trial_no_in_block + 1)).read().astype(int)
        ev_u0_non_ovlp = ev_u0[ovlp_u0 == 0]
        ev_u1_non_ovlp = ev_u1[ovlp_u1 == 0]
    arc_gt.close()
    del arc_gt

    # get events, epochs and spikes
    events = sp.concatenate((ev_u0, ev_u1))
    events.sort()
    epochs = epochs_from_spiketrain(events, get_cut(TF, off=CUTOFFSET))
    if do_ovlps is False:
        events_non_ovlp = sp.concatenate((ev_u0_non_ovlp, ev_u1_non_ovlp))
        events_non_ovlp.sort()
        if events_non_ovlp.size == 0:
            spikes = sp.zeros((0, TF * 4))
        else:
            epochs_non_ovlp = epochs_from_spiketrain(events_non_ovlp, get_cut(TF, off=CUTOFFSET))
            spikes = extract_spikes(sweep_data, epochs_non_ovlp)
        rval.append(events_non_ovlp)
    else:
        spikes = extract_spikes(sweep_data, epochs)
        rval.append(events)
    rval.append(spikes)

    # get covariance matrix
    cov_est = TimeSeriesCovE(tf_max=TF, nc=4)
    cov_est.new_chan_set((0, 1, 2, 3))
    cov_est.update(sweep_data, epochs=invert_epochs(epochs, end=1000000))
    rval.append(cov_est.get_cmx(tf=TF, chan_set=(0, 1, 2, 3)))
    rval.append(1000000 - events.size * TF)

    # return
    print '..done!'
    return rval


def atf_block_clustering(atf_path,
                     save_path,
                     exp,
                     block,
                     do_ovlps=True,
                     do_plots=False):
    """write a block wise nas file

    :Parameters:
        atf_path : str
            path to the directory where the atf files reside
        save_path : str
            path to the directory where the nas and gf file save to
        exp : str
            the experiment to convert
        block : str
            the block to convert
        do_plots : bool
            plot detection results as a stacked waveform plot
            Default=False
        do_intra : bool
            if true, detect from the groundtruth, else detect with conventional
            spikedetection
            Default=True
        do_ovlps : bool
            if true and do_intra is true, include overlaps, esle exclude overlaps
    """

    # checks
    if exp not in EXP_DICT:
        raise ValueError('unknown experiment: %s' % exp)
    if block not in EXP_DICT[exp]:
        raise ValueError('unknown block: %s, in experiment: %s' % (block, exp))

    # inits
    init = {}

    # get per trial initializations
    for fname in EXP_DICT[exp][block]:
        init[fname] = initialization(osp.join(atf_path, exp, fname),
                                     do_ovlps=do_ovlps)
    init_idx = {}
    ii = 0
    cov = sp.zeros((TF * 4, TF * 4))
    spikes = []
    nsamples = 0
    for fname in init:
        cov += init[fname][2] * init[fname][3]
        nsamples += init[fname][3]
        spikes.append(init[fname][1])
        init_idx[fname] = ii, ii + init[fname][0].size
        ii += init[fname][0].size
    cov /= nsamples
    spikes = sp.concatenate(spikes)

    # do clustering
    PRE_ali = AlignmentNode(max_tau=int(TF / 2.0), debug=True)
    PRE_whi = PrewhiteningNode(cov)
    PRE_pca = PCANode(output_dim=.5)
    clus_data = PRE_ali(spikes)
    if do_plots:
        waveforms(
            spikes,
            filename=osp.join(save_path, 'wf_%s_%s' % (exp, block)),
            show=False
        )
    clus_data = PRE_whi(clus_data)
    clus_data = PRE_pca(clus_data)
    print PRE_pca.get_explained_variance(), '% variance @', PRE_pca.output_dim, 'components'
    CLUS = HomoscedasticClusteringNode(clus_type='kmeans', crange=[2],
                                       maxiter=256, repeats=16)
    CLUS(clus_data)

    # create gdfs
    for fname in EXP_DICT[exp][block]:
        labels = CLUS.labels[init_idx[fname][0]:init_idx[fname][1]]
        gdf = {}
        for i in xrange(int(CLUS.labels.max() + 1)):
            gdf[i] = init[fname][0][labels == i]
        GdfFile.write_gdf(osp.join(save_path, ''.join([fname[:-4], '.gdf'])), gdf)


##---MAIN

if __name__ == '__main__':

    # switches
    RANGE = 'ALL#'
    DO_OVLPS = True
    DO_PLOTS = True

    # paths
    atf_path = '/home/phil/Data/Alle/atfdata/'
    save_path = '/home/phil/Data/Alle/clustering_new/'

    # run analysis
    print 'STARTING'

    if RANGE == 'ALL':
        for exp in sorted(EXP_DICT):
            print 'EXP:', exp
            for block in sorted(EXP_DICT[exp]):
                print 'BLOCK:', block
                atf_block_clustering(atf_path, save_path, exp, block,
                                     do_ovlps=DO_OVLPS, do_plots=DO_PLOTS)
    else:
        atf_block_clustering(atf_path, save_path, '19022008', 'C',
                             do_ovlps=DO_OVLPS, do_plots=DO_PLOTS)

    # end
    print 'ALL DONE!'
