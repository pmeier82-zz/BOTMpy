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


"""
Created on Sat Feb 05 19:34:58 2011

@author: phil
"""


## imports
import os
import scipy as sp
from util import DATAPATH
from tables import openFile

### elaborate on pixel index set
#print 'finding index set'
#x_coords = []
#y_coords = []
#d_len = []
#for f in os.listdir(DATAPATH):
#    if not f.endswith('1x1data'):
#        continue
#    x,y = tuple(map(int, f.split('.')[3].split('-')))
#    if x not in x_coords:
#        x_coords.append(x)
#    if y not in y_coords:
#        y_coords.append(y)
#    print x, y
#X_ = sp.genfromtxt(os.path.join(DATAPATH, f), skip_header=1, usecols=(1,))
#d_len = X_.size
#x_coords.sort()
#y_coords.sort()
arc = openFile(os.path.join(DATAPATH, 'cmos_data.h5'), 'r+')
#arc.createArray(arc.root, 'x_coords', x_coords)
#arc.createArray(arc.root, 'y_coords', y_coords)
#
#print
#print 'RESULTS: index set'
#print 'data length', d_len
#print len(x_coords), x_coords
#print len(y_coords), y_coords
#print

## produce scipy array of data
X = arc.getNode('/data')
#X = arc.createArray(arc.root, 'data', sp.zeros((d_len, len(y_coords) * len(x_coords)), dtype=sp.float32))
#X = sp.zeros((d_len, len(y_coords) * len(x_coords)), dtype=sp.float32)

#i=0
#for x in x_coords:
#    for y in y_coords:
#        print x, y
#        fname = os.path.join(DATAPATH, 'data00042.nfx.cpd.%d-%d.1x1data' % (x,y))
#        X[:, i] = sp.genfromtxt(fname, skip_header=1, usecols=(1,), dtype=sp.float32)
#        i += 1
#arc.flush()
    
## chunked covariance estimation
slice_size = 1000
print 'starting to estimate cmx'
cmx = arc.createArray(arc.root, 'R0', sp.zeros((X.shape[1], X.shape[1]), sp.float32))
#cmx = arc.getNode('/R0')
#cmx[:] = 0.0
#cmx = sp.zeros((X.shape[1], X.shape[1]), sp.float32)
sidx = 0
cdiv = 0.0
while sidx < X.shape[0]:
    if sidx + slice_size >= X.shape[0]:
        slice_size = X.shape[0] - sidx
    print '[%s:%s]' % (sidx,sidx+slice_size),
    my_slice = X[sidx:sidx+slice_size,:]
    cmx += sp.cov(my_slice.T)
    cdiv += my_slice.shape[0] / float(slice_size)
    sidx += slice_size
    del my_slice
    arc.flush()
print
print 'estimated cmx, correcting for %d' % cdiv
cmx /= cdiv
print 'estimated cmx'

## close hdf5 file
arc.close()
