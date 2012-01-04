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


"""convert a ATF file (from Hendrik Alle) to NAS file (Chen Sorter)"""


##---IMPORTS

# builtins
from plot import P
from os import path as osp
import scipy as N
# own imports
from common import extract_spikes
from common.datafile import AtfFile, GdfFile, NasFile
from nodes import SDIntraNode, SDMteoNode
from util import *


##---CONSTANTS

# define the cutting window for events in ms
cutleft = 1.0
cutright = 1.0
#########################################################
# if you touch stuff below your computer will explode!! #
#########################################################
CUT = (N.ceil(cutleft / SSTEP), N.ceil(cutright / SSTEP))
CUTmusec = (CUT[0] * SSTEP * 1e3, (CUT[1] + 1) * SSTEP * 1e3 + 1)
TF = int(N.sum(CUT) + 1) # spike windows in samples
TRIAL_OFFSET_MS = 50000


##---FUNCTIONS

def detect_atf_events(fname_in,
                      trial_no_in_block=0,
                      delta_tau=MAX_OVERLAP,
                      do_intra=True,
                      do_ovlps=True,
                      do_plots=False):
    """detect events from atf file
    
    :Parameters:
        fname_in : str
            path to an existing .atf file to read from
        trial_no_in_block : int
            offset for the timing in samples
        delta_tau : int
            events that have a distance of <= delta_tau are classified as
            overlaps, see do_ovlps.
            Default=MAX_OVERLAP
        do_intra : bool
            if true, use the intracellular data (groundtruth)to find the spiketrains for the
            units, this will let you choose if you want to include or exclude overlaps via
            the do_ovlps parameter. Eles use the extracellular data to find the spiketrains
            for the units, the do_ovlps parameter will then be ignored.
            Default=True
        do_ovlps : bool
            if true, include overlaps in the events returned, else exclude them
            Default=True
        do_plots : bool
            if true, plot stacked waveforms of found spikes
            Default=False
    :Returns:
        (dict, dict) : dict with the (spiketime, waveform) tuples of both units labeled and
        a dict with the groundtruth in gdf
        structure. 
    """

    print 'starting file', osp.basename(fname_in),

    # inits
    rval = {}

    # read in from atf file
    arc_in = AtfFile(fname_in)
    gtrth_data = arc_in.get_data(mode='device', item=0)
    sweep_data = arc_in.get_data(mode='device', item=1)
    sweep_data -= sweep_data.mean(0)
    sweep_time = arc_in.header.sweep_times
    arc_in.close()
    del arc_in

    # detection
    if do_intra:
        ndt = 2
        DT = [SDIntraNode(tf=TF, cut=CUT, extract=False) for i in xrange(ndt)]
        ev = [None for i in xrange(ndt)]
        for i in [0, 1]:
            ev[i] = DT[i](N.atleast_2d(gtrth_data[:, i]).T)
        # overlap exclusion
        if not do_ovlps:
            non_ovlp = [N.ones_like(ev[i]) for i in xrange(ndt)]
            idx0 = idx1 = 0
            while idx0 < len(ev[0]) and idx1 < len(ev[1]):
                # check overlap
                if max(ev[0][idx0], ev[1][idx1]) - min(ev[0][idx0], ev[1][idx1]) < delta_tau:
                    non_ovlp[0][idx0] = 0
                    non_ovlp[1][idx1] = 0
                    idx0 += 1
                    idx1 += 1
                else:
                    if ev[0][idx0] <= ev[1][idx1]:
                        idx0 += 1
                    else:
                        idx1 += 1
            for i in xrange(ndt):
                ev[i] = ev[i][non_ovlp[i] == 1]
                DT[i].events = ev[i]
    else:
        ndt = 1
        DT = [SDMteoNode(tf=TF, cut=CUT, extract=False)]
        ev = [DT(sweep_data)]

    # generate epochs and spikes
    ep = [DT[i].get_epochs(invert=False, merge=False, cut=CUT) for i in xrange(ndt)]
    sp = [extract_spikes(sweep_data, ep[i]) for i in xrange(ndt)]

    # plotting
    if do_plots:
        fig = P.figure()
        ax = []
        for i in xrange(4):
            ax.append(fig.add_subplot(221 + i))

    for u in xrange(ndt):
        # per UNIT LOOP
        if u not in rval:
            rval[u] = []
        for s in xrange(ev[u].size):
            # per SPIKE LOOP
            # timing info - sweeps are 10000 sample long,
            # we know sweep start times and the time-frame of one sample
            spike_time = sweep_time[ev[u][s] / 10000] + (ev[u][s] % 10000) * SSTEP
            # add the trial offset
            spike_time += trial_no_in_block * TRIAL_OFFSET_MS
            # write nas data
            rval[u].append((spike_time, sp[u][s]))
            # plotting
            if do_plots:
                for i in xrange(4):
                    ax[i].plot(sp[u][s, i * TF:(i + 1) * TF], c='k')

    # plotting
    if do_plots:
        P.show()

    # return
    print '..done!'
    return rval


def atf_block_to_nas(atf_path,
                     save_path,
                     exp,
                     block,
                     do_intra=True,
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
    nasfile_name = osp.join(save_path, ''.join([exp, '-', block, '.nas']))
    nasfile = NasFile(nasfile_name)
    nasfile.write_header(4, SRATE, CUTmusec[0], CUTmusec[1])

    # get going
    trial_no_in_block = 0
    for fname in EXP_DICT[exp][block]:
        # do detection
        nas_d = detect_atf_events(
            osp.join(atf_path, exp, fname),
            trial_no_in_block=trial_no_in_block,
            do_intra=do_intra,
            do_ovlps=do_ovlps,
            do_plots=do_plots
        )
        # write to nas and gdf
        for u in nas_d:
            for sp_t, sp_d in nas_d[u]:
                nasfile.write_row(
                    tetr=1,
                    unit=u + 1,
                    trial=1,
                    time=sp_t, # timing in ms from block offset
                    data=sp_d, # waveform data
                )
        trial_no_in_block += 1

    # sort gdf
    nasfile.close()
    gdf_d = NasFile.get_gdf(nasfile_name)
    gdf_d[:, 1] = gdf_d[:, 1] * int(SRATE * 0.001)
    gdf_d = GdfFile.convert_matrix_to_dict(gdf_d)
    GdfFile.write_gdf(osp.join(save_path, ''.join([exp, '-', block, '.gdf'])), gdf_d)


##---MAIN

if __name__ == '__main__':

    # switches
    RANGE = 'ALL'
    DO_OVERLAPS = False

    # paths
    atf_path = '/home/phil/Data/Alle/atfdata/'
    nas_path = '/home/phil/Data/Alle/nasdata/blockweise_ohne_overlaps/'

    # run analysis
    print 'STARTING'

    if RANGE == 'ALL':
        for exp in EXP_DICT:
            print 'EXP:', exp
            for block in EXP_DICT[exp]:
                print 'BLOCK:', block
                atf_block_to_nas(atf_path, nas_path, exp, block, do_ovlps=DO_OVERLAPS)
    else:
        nas_path += '1/'
        atf_block_to_nas(atf_path, nas_path, '19022008', 'A')

    # end
    print 'ALL DONE!'
