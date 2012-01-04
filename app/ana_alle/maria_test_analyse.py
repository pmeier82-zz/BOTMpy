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


from common.datafile import GdfFile, WriFile, NasFile
import numpy as np
from common import align_spike_trains
import plot
#from pydevd import set_pm_excepthook
#set_pm_excepthook()
# Define the folders and files
#org_gdf_file = 'C:/SVN/Datenanalyse/Alle/0_tetrode.gdf'
#maria_wri_file = 'C:/SVN/Datenanalyse/Alle/write_test000.wri'
#maria_gdf_file = 'C:/SVN/Datenanalyse/Alle/write_test000.gdf'
org_gdf_file = 'C:/Data/Alle/wri_wo_overlaps/19022008-B.gdf'
maria_wri_file = 'C:/Data/Alle/wri_wo_overlaps/19022008-B000.wri'
maria_gdf_file = 'C:/Data/Alle/wri_wo_overlaps/19022008-B_maria.gdf'

# Get GDF from NAS File
nas_file = 'C:/Data/Alle/wri_w_overlaps/19022008-B.nas'
nas_gdf = NasFile.get_gdf(nas_file)
# Change to the right time resolution
nas_gdf[:, 1] = nas_gdf[:, 1] * 33
nas_gdf = GdfFile.convert_matrix_to_dict(nas_gdf)
# Save gdf just in case
GdfFile.write_gdf(nas_file + '.gdf', nas_gdf)

# Do not read the original gdf, since the times are not right
#org_gdf = GdfFile.read_gdf(org_gdf_file)
org_gdf = nas_gdf
maria_gdf = WriFile(maria_wri_file).get_data()

# Save gdf from wri file, just in case
GdfFile.write_gdf(maria_gdf_file, maria_gdf)

# Do the alignment, maybe we have to increase maxjitter or maxshift?
ret = align_spike_trains(org_gdf, maria_gdf, maxshift=20, maxjitter=66, maxoverlapdistance=45)
from common import print_nice_table
print_nice_table(ret)

# Make an insanely slow plot
from plot import P
fig = P.figure(facecolor='white')
# TODO: Plot only beginning of spike trains!
#plot.spike_trains(org_gdf, spiketrains2=maria_gdf, alignment=ret['alignment'], label1=ret['GL'], label2=ret['EL'], plot_handle=fig, samples_per_second=33333)
plot.spike_trains(org_gdf, spiketrains2=maria_gdf, plot_handle=fig, samples_per_second=33000)
P.show()

def make_ana_block():
    pass
#
