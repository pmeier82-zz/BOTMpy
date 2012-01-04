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


"""validate a sorting give a benchmark ground truth file and a sorting archive"""
__docformat__ = 'restructuredtext'


##--- IMPORTS

from util import *

from tables import openFile
from os import path as osp
from common import align_spike_trains

import scipy as N


##---FUNCTIONS

def validate_sorting():
    """read in data from benchmark file and sorting file, then do a validation"""

    rval = False, None

    try:
        # check input archives
        if not osp.exists(osp.join(WORKING_PATH, INPUT_ARC)):
            raise IOError('benchmark archive not found')
        if not osp.exists(osp.join(WORKING_PATH, OUTPUT_ARC)):
            raise IOError('sorting archive not found')
        arc_gt = openFile(osp.join(WORKING_PATH, INPUT_ARC), 'r')
        arc_so = openFile(osp.join(WORKING_PATH, OUTPUT_ARC), 'r')

        # bring both spike trains in the format we need for align_spike_trains
        train_gt = {}
        for unit_gt in arc_gt.getNode('/groundtruth'):
            train_gt[unit_gt._v_name] = unit_gt.train.read()
        in_dict_change_list_to_array(train_gt)
        train_so = {}
        for unit_so in arc_so.root:
            train_so[unit_so._v_name] = unit_so.read()
        in_dict_change_list_to_array(train_so)

        # do validation
        validation = align_spike_trains(
            train_gt,
            train_so,
            maxshift=TF,
            maxjitter=10,
            maxoverlapdistance=TF
        )
        rval = True, validation

        # do plotting
        from plot import P, spike_trains
        fig = P.figure(facecolor='white')
        spike_trains(
            train_gt,
            spiketrains2=train_so,
            alignment=validation['alignment'],
            label1=validation['GL'],
            label2=validation['EL'],
            plot_handle=fig,
            samples_per_second=16000
        )
    except Exception, ex:
        rval = False, ex
    finally:
        try:
            arc_gt.close()
        except:
            pass
        try:
            arc_so.close()
        except:
            pass
        return rval


def in_dict_change_list_to_array(d):
    for key in d:
        if isinstance(d[key], (list, tuple)):
            d[key] = N.asarray(d[key])


##--- MAIN

if __name__ == '__main__':

    rval = validate_sorting()
    if rval[0] is True:
        arc = openFile('validation.h5', 'w')
        ppt = N.asarray(rval[1]['table'], dtype=str)
        arc.createArray(arc.root, 'validation_table', ppt)
        arc.close()
    else:
        print '## ERROR ##'
        print rval[1]
