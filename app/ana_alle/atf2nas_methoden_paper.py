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


"""convert a ATF file (from Hendrik Alle) to NAS file (Chen Sorter)

this version uses the unified groundtruth for the methoden paper
"""


##---IMPORTS

# builtins
from plot import waveforms
from os import path as osp
import scipy as sp
from common import (epochs_from_spiketrain, extract_spikes, AtfFile, GdfFile,
                    NasFile)
from tables import openFile
from util import *


##---CONSTANTS

# define the cutting window for events in ms
cutleft = 1.4
cutright = 1.4
cutoffset = 20
#########################################################
# if you touch stuff below your computer will explode!! #
#########################################################
CUT = int(sp.ceil(cutleft / SSTEP) - cutoffset), int(sp.ceil(cutright / SSTEP) + cutoffset)
CUTmusec = (CUT[0] * SSTEP * 1e3, (CUT[1] + 1) * SSTEP * 1e3 + 1)
TF = int(sp.sum(CUT) + 1) # spike windows in samples
TRIAL_OFFSET_MS = 50000
GTPATH = '/home/phil/Dropbox/MunkShare/Methoden Paper/AuswertungAlleDaten/GroundTruth'


##---FUNCTIONS

def detect_atf_events(fname_in,
                         trial_no_in_block=0,
                         do_ovlps=True,
                         do_plots=False):
    """detect events from atf file
    
    :Parameters:
        fname_in : str
            path to an existing .atf file to read from
        trial_no_in_block : int
            offset for the timing in samples
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
    rval = []

    # read in from atf file
    arc_in = AtfFile(fname_in)
    sweep_data = arc_in.get_data(mode='device', item=1)
    sweep_data -= sweep_data.mean(0)
    sweep_time = arc_in.header.sweep_times
    arc_in.close()
    del arc_in

    # get groundtruth
    exp, block = get_info_for_fname(osp.basename(fname_in))
    arc_gt = openFile(osp.join(GTPATH, '%s%s_intracSpDetResults.h5' % (exp, block)), 'r')
    gt = arc_gt.getNode('/singleFileResults/%05d/gdf' %
                        int(trial_no_in_block + 1)).read().astype(int)

    # get events
    ev_u0 = gt[1][gt[0] == 1]
    ev_u1 = gt[1][gt[0] == 2]
    if do_ovlps is False:
        ovlp_u0 = arc_gt.getNode('/singleFileResults/%05d/O/00001' %
                                 int(trial_no_in_block + 1)).read().astype(int)
        ovlp_u1 = arc_gt.getNode('/singleFileResults/%05d/O/00002' %
                                 int(trial_no_in_block + 1)).read().astype(int)
        ev_u0 = ev_u0[ovlp_u0 == 0]
        ev_u1 = ev_u1[ovlp_u1 == 0]
    arc_gt.close()
    del arc_gt

    # get events, epochs and spikes
    events = sp.concatenate((ev_u0, ev_u1))
    events.sort()
    epochs = epochs_from_spiketrain(events, CUT)
    spikes = extract_spikes(sweep_data, epochs)

    for s in xrange(events.size):
        # per SPIKE LOOP
        # timing info - sweeps are 10000 sample long,
        # we know sweep start times and the time-frame of one sample
        spike_time = sweep_time[events[s] / 10000] + (events[s] % 10000) * SSTEP
        # add the trial offset
        spike_time += trial_no_in_block * TRIAL_OFFSET_MS
        # write nas data
        rval.append((spike_time, spikes[s]))
    # plotting
    if do_plots:
        waveforms(spikes)

    # return
    print '..done!'
    return rval


def atf_block_to_nas(atf_path,
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
            do_ovlps=do_ovlps,
            do_plots=do_plots
        )
        # write to nas and gdf
        for sp_t, sp_d in nas_d:
            nasfile.write_row(
                tetr=1,
                unit=1,
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
    nas_path = '/home/phil/Data/Alle/nasdata/blockweise_ohne_overlaps'

    # run analysis
    print 'STARTING'

    if RANGE == 'ALL':
        for exp in EXP_DICT:
            if exp == 'test':
                continue
            print 'EXP:', exp
            for block in EXP_DICT[exp]:
                if block == 'test':
                    continue
                print 'BLOCK:', block
                atf_block_to_nas(atf_path, nas_path, exp, block, do_ovlps=DO_OVERLAPS)
    else:
        atf_block_to_nas(atf_path, nas_path, '19022008', 'A', do_ovlps=DO_OVERLAPS)

    # end
    print 'ALL DONE!'
