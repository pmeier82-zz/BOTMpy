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


from common import (GdfFile, WriFile, NasFile, align_spike_trains,
                    print_nice_table, csv_from_analysis)
from os.path import join
import plot

#from pydevd import set_pm_excepthook
#set_pm_excepthook()

#Philipps Pfade
GDFPATH = '/home/phil/Data/Alle/nasdata/new_ana'
WRIPATH = '/home/phil/Data/Alle/wri'

# Pfade Mizar MIT Overlaps
GDFPATH = 'X:/Alle/nas_blockweise/'
WRIPATH = 'X:/Alle/Maria/wri_with_overlaps_V1/'
WRIPATH = 'X:/Alle/Maria/wri_with_overlaps_V2/'

# Pfade Mizar OHNE Overlaps
GDFPATH = 'X:/Alle/nas_blockweise_ohne_overlaps/'
WRIPATH = 'X:/Alle/Maria/wri_without_overlaps/'

def do_analysis_block(exp, block, do_plot=False, show_plot=False, save_plot=False):
    print '\nANA %s BLOCK: %s' % (exp, block)

    # filenames
    gdf_name = join(GDFPATH, '%s-%s.gdf' % (exp, block))
    wri_name = join(WRIPATH, '%s-%s000.wri' % (exp, block))
    pic_name = join(WRIPATH, '%s-%s000.png' % (exp, block))
    nas_name = join(GDFPATH, '%s-%s.nas' % (exp, block))
    csv_name = join(WRIPATH, '%s-%s000.csv' % (exp, block))

    ##################################
    # READ THE NAS FILE TO GET THE GT GDF
    nas_gdf = NasFile.get_gdf(nas_name)
    # Change to the right time resolution
    nas_gdf[:, 1] = nas_gdf[:, 1] * 33
    nas_gdf = GdfFile.convert_matrix_to_dict(nas_gdf)
    # Save gdf just in case
    GdfFile.write_gdf(nas_name + '.gdf', nas_gdf)
    ##################################

    ##################################
    # READ THE GDF FILE TO GET THE GT GDF
    # this should be the same as the NAS file !!!!
    gdf_data = GdfFile.read_gdf(gdf_name)
    wri_data = WriFile(wri_name).get_data()

    wri_gdf_name = join(GDFPATH, '%s-%s_wri.gdf' % (exp, block))
    GdfFile.write_gdf(wri_gdf_name, wri_data)
    ##################################

    GT_gdf = nas_gdf

#    for key in GT_gdf.keys():
#        GT_gdf[key] = GT_gdf[key][:100]
# 
#    for key in wri_data.keys():
#        wri_data[key] = wri_data[key][:100]

    # DO NOT USE THE GDF FROM THE GDF FILE BUT BUILD ONE FROM THE NAS !!!
    #GT_gdf = gdf_data

    # do the alignment
    ali = align_spike_trains(GT_gdf,
                             wri_data,
                             maxshift=60,
                             maxjitter=66,
                             maxoverlapdistance=45)

    # Print
    print_nice_table(ali)
    csv_str = csv_from_analysis(ali, header=True)
    fh = open(csv_name, 'w')
    fh.write(csv_str)
    fh.close()

    # plot stuff
    if do_plot:
        print "Plotting..."
        fig = plot.P.figure(facecolor='white')
        plot.spike_trains(GT_gdf,
                          spiketrains2=wri_data,
#                          alignment=ali['alignment'],
                          label1=ali['GL'],
                          label2=ali['EL'],
                          plot_handle=fig,
#                          samples_per_second=33000,
                          show=show_plot)
        print "done."
        if save_plot:
            print "Saving Plot: %s", pic_name
            fig.savefig(pic_name)
    # return stuff
    return ali, gdf_data, wri_data, csv_str

##--- MAIN
if __name__ == '__main__':
    from util import EXP_DICT


    RANGE = 'ALL'
    WITH_PLOTS = True

    # start
    print 'STARTING'
    if RANGE == 'ALL':
        for exp in EXP_DICT:
            for block in EXP_DICT[exp]:
                ali, gt, ana, csv_str = do_analysis_block(exp, block, do_plot=WITH_PLOTS, save_plot=True)
                fh = open('%sauswertung.csv' % (WRIPATH), 'w+')
                fh.write('%s, %s\n' % (exp, block))
                fh.write(csv_str)
                fh.close()
    else:
        # GDFPATH += '/1'
        exp = '24022009'
        block = 'A'
        ali, gt, ana, csv_str = do_analysis_block(exp, block, do_plot=WITH_PLOTS, show_plot=True)
        fh = open('%sauswertung.csv' % (WRIPATH), 'w+')
        fh.write('%s, %s\n' % (exp, block))
        fh.write(csv_str)
        fh.close()
    print 'ALL DONE'
