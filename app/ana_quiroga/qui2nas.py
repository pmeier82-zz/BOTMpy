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
import scipy as sp, scipy.io as sp_io
# own imports
from common import extract_spikes
from common.datafile import AtfFile, GdfFile, NasFile
from nodes import AlignmentNode, ThresholdDetectorNode
from util import *


##---CONSTANTS

CUT = (10, 84)
TFcut = 10 + 84
cut2_start = 18
TF = 64


##---FUNCTIONS

def qui2nas(fname_in,
            fname_out,
            delta_tau=MAX_OVERLAP,
            do_gtrth=True,
            do_ovlps=True,
            do_plots=False):
    """detect events from atf file
    
    :Parameters:
        fname_in : str
            path to an existing .mat file to read from
        fname_out : str
            path to an non existing location to store the resulting nas and gdf
        delta_tau : int
            events that have a distance of <= delta_tau are classified as
            overlaps, see do_ovlps.
            Default=MAX_OVERLAP
        do_gtrth : bool
            if true, use the ground truth from the file, else use spike
            detection.
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

    # read in from atf file
    arc_in = sp_io.loadmat(fname_in)
    data = arc_in['data'].T
    sstep = arc_in['samplingInterval'][0][0]

    # detection
    if do_gtrth:
        evts = arc_in['spike_times'][0][0][0]
        lbls = arc_in['spike_class'][0][0][0]
    else:
        pass
    del arc_in

    # use bigger TF to cut the spikes. this allows us to cut the spikes again
    # after alignment to avoid having zeros at the end of the spikes
    # generate epochs and spikes
    TDN = ThresholdDetectorNode(tf=TFcut, cut=CUT)
    TDN.train(data)
    TDN.stop_training()
    TDN.events = evts
    spks = TDN.get_extracted_events()
    # alignment on mean spike
    AN = AlignmentNode(nchan=1)
    spks = AN(spks)

    # cut spikes a second time, this time after alignment
    spks = spks[:, cut2_start:cut2_start + TF];

    # convert to int
    spks = (spks * 256).astype(sp.int32)

    # plotting
    if do_plots:
        fig = P.figure()
        ax = fig.add_subplot(111)

    # build nas dataset
    nas_d = []
    for s in xrange(spks.shape[0]):
        # timing info
        spike_time = evts[s] * sstep
        # spike waveform data
        nas_d.append((spike_time, spks[s]))
        # plotting
        if do_plots:
            ax.plot(spks[s], c='k')

    # plotting
    if do_plots:
        P.show()

    # inits
    nasfile = NasFile(fname_out)
    nasfile.write_header(1,
                         1000 / sstep,
                         sp.floor(TF / 2) * sstep * 1000,
                         sp.floor(TF / 2) * sstep * 1000)

    for s in xrange(len(nas_d)):
        nasfile.write_row(
            tetr=1,
            unit=lbls[s] + 1,
            trial=1,
            time=nas_d[s][0],
            data=nas_d[s][1],
        )

    # sort gdf
    nasfile.close()
    gdf_d = NasFile.get_gdf(fname_out)
    gdf_d[:, 1] = gdf_d[:, 1] * int(1000 / sstep * 0.001)
    gdf_d = GdfFile.convert_matrix_to_dict(gdf_d)
    GdfFile.write_gdf(''.join([fname_out, '.gdf']), gdf_d)


##---MAIN

if __name__ == '__main__':

    # switches
    RANGE = 'ALL'
    DO_OVERLAPS = False

    # run analysis
    print 'STARTING'

    if RANGE == 'ALL':
        for exp in EXP_DICT:
            print 'EXP:', exp
            for npart in EXP_DICT[exp]:
                fname = 'C_' + exp + '_' + npart
                qui2nas(osp.join(INPATH, fname + '.mat'),
                        osp.join(OUTPATH, fname + '.nas'),
                        do_ovlps=DO_OVERLAPS,
                        do_gtrth=True,
                        do_plots=False)
    else:
        fname = 'C_Easy1_noise005'
        qui2nas(osp.join(INPATH, fname + '.mat'),
                osp.join(OUTPATH, fname + '.nas'),
                do_ovlps=DO_OVERLAPS,
                do_gtrth=True,
                do_plots=True)

    # end
    print 'ALL DONE!'
