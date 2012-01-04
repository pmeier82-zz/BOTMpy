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
Created on Sat Feb 05 20:35:55 2011

@author: phil
"""


## imports

from util import DATAPATH
from tables import openFile
from os import path as osp
import scipy as sp
from plot import P


## constants

slice_size = 500


## load data

arc = openFile(osp.join(DATAPATH, 'cmos_data.h5'), 'r')
data = arc.getNode('/data').read()
arc.close()
del arc


## chunking of the data

print 'starting to estimate cmx'
cmx = sp.zeros((data.shape[0], data.shape[0]))

sidx = 0
cdiv = 0
while sidx < data.shape[1]:
    print '[%s:%s]' % (sidx,sidx+slice_size),
    my_slice = data[:,sidx:sidx+slice_size]
    cmx += sp.cov(my_slice)
    cdiv += my_slice.shape[1] / float(slice_size)
    sidx += slice_size
print
print 'estimated cmx, correcting for %d' % cdiv
cmx /= cdiv
print 'estimated cmx'


## show stuff

del my_slice, data, cdiv, sidx
P.matshow(cmx)
P.figure()
P.plot(sp.diag(cmx))
P.show()
