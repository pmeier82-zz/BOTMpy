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


import scipy as sp
from common import GdfFile, WriFile
from common.spike_train_correlation import align_spike_trains, print_nice_table
import plot
import os.path as osp
from util import *


##---CONSTANTS

GDFPATH = '/home/phil/Data/Alle/nasdata/new_ana'
WRIPATH = '/home/phil/Data/Alle/wri'


##---FUNCTIONS

def do_analysis(exp, block, save_wri_gdf=False):

    # filenames
    gdf_name = osp.join(GDFPATH, '%s-%s.gdf' % (exp, block))
    wri_name = osp.join(WRIPATH, '%s-%s000.wri' % (exp, block))

    # read in data
    gdf_data = GdfFile.read_gdf(gdf_name)
    wri_data = WriFile(wri_name).get_data()
    if save_wri_gdf:
        wri_gdf_name = osp.join(GDFPATH, '%s-%s_wri.gdf' % (exp, block))
        GdfFile.write_gdf(wri_gdf_name, wri_data)

    # do the alignment
    del wri_data['B']
    ali = align_spike_trains(gdf_data,
                             wri_data,
                             maxshift=100,
                             maxjitter=100,
                             maxoverlapdistance=69)

    # return stuff
    return ali, gdf_data, wri_data

def present_analysis(ali, gt_data, ana_data, do_plot=False, show_plot=False, save_plot=False):
    print_nice_table(ali)

    # plot stuff
    if do_plot:
        fig = plot.P.figure(facecolor='white')
        plot.spike_trains(gt_data,
                          spiketrains2=ana_data,
                          alignment=ali['alignment'],
                          label1=ali['GL'],
                          label2=ali['EL'],
                          plot_handle=fig,
                          samples_per_second=33000,
                          show=show_plot)
        if save_plot is not False:
            fig.savefig()


##---MAIN

if __name__ == '__main__':

    # switches
    WITH_PLOTS = True
    RANGE = 'ALL#'

    # start
    print 'STARTING'
    if RANGE == 'ALL':
        for exp in EXP_DICT:
            if exp == 'test':
                continue
            print '\nANA %s' % exp
            for block in EXP_DICT[exp]:
                print 'BLOCK:', block
                ali, gt, ana = do_analysis(exp, block)
                present_analysis(ali, gt, ana, do_plot=WITH_PLOTS)
    else:
        # GDFPATH += '/1'
        ali, gt, ana = do_analysis('19022008', 'B')
        present_analysis(ali, gt, ana, do_plot=WITH_PLOTS, show_plot=True)
    print 'ALL DONE'
