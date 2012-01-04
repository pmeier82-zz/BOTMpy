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


"""simple sorting script that generates spiketrains, given data, noise and
waveforms, saveingto hdf5 archive"""
__docformat__ = 'restructuredtext'


##--- IMPORTS

from util import *

from nodes import FSSNode
from common import TimeSeriesCovE, ten2vec
from tables import openFile

import os
from os import path as osp

import scipy as N


##---FUNCTIONS

def sorting():
    """start the sorting in the given directory"""

    rval = False, None

    try:
        # check if archives are present
        if not osp.exists(WORKING_PATH):
            raise IOError('%s does not exist')
        if not osp.exists(osp.join(WORKING_PATH, INPUT_ARC)):
            raise IOError('input arc does not exist')

        # open and read input arc
        inp_arc = openFile(osp.join(WORKING_PATH, INPUT_ARC), 'r')
        data_ana = inp_arc.getNode('/data').read()
        data_noise = inp_arc.getNode('/noise').read()
        data_wf = []
        for unit in inp_arc.getNode('/groundtruth'):
            data_wf.append(unit.waveform.read()[:TF, :])
        data_wf = ten2vec(N.asarray(data_wf))

        # sort data
        cov_est = TimeSeriesCovE(tf=TF)
        cov_est.update(data_noise)
        sorter = FSSNode(data_wf, cov_est.get_icmx(), cov_est.get_icmx())
        sorting = sorter(data_ana)

        # save sorting
        out_arc = openFile(osp.join(WORKING_PATH, OUTPUT_ARC), 'w')
        for unit in sorting:
            out_arc.createArray(out_arc.root, 'unit_%s' % unit, sorting[unit])
            out_arc.flush()

        # return value
        rval = True, 'ALL DONE'

    except Exception, ex:
        rval = False, ex
    finally:
        try:
            inp_arc.close()
        except:
            pass
        try:
            out_arc.close()
        except:
            pass
        return rval


##--- MAIN

if __name__ == '__main__':

    rval = sorting()
    print 'rval was:', rval
