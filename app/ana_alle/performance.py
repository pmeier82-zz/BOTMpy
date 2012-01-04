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


from nodes import ThresholdDetectorNode
import scipy as N
import os
import os.path as osp
from tables import openFile
from util import *
from common import align_spike_trains


##---FUNCTIONS

def performance_analysis(exp='19022008', trial=0, save=False):
    """produce a spike sorting performance analysis per trial"""

    print 'perf_ana(trial): %s %04d' % (exp, trial)

    # get data from archive
    block = get_block_for_trial(exp, trial)
    arc = openFile(osp.join(HDFPATH, '%s_fss.h5' % exp), 'r')
    GT = {
        'gt0' : arc.getNode('/%s_%04d/gt_unit0' % (exp, trial)).read(),
        'gt1' : arc.getNode('/%s_%04d/gt_unit1' % (exp, trial)).read(),
    }
    fouts = arc.getNode('/%s_%04d/ss_fouts' % (exp, trial)).read()
    tf = int(arc.getNode('/tf').read())
    arc.close()
    del arc

    # get sorting
    OF = ThresholdDetectorNode(
        N.zeros(2),
        N.eye(2),
        N.eye(2),
        th_fac=EXP_THRESHOLD_DICT[exp][block],
        tf=tf
    )
    # do for unit0
    OF.energy = N.atleast_2d(fouts[0].copy()).T
    OF.size, OF.nchan = OF.energy.shape
    ss0 = OF._execute(extract=False)
    # do for unit1
    OF.energy = N.atleast_2d(fouts[1].copy()).T
    OF.size, OF.nchan = OF.energy.shape
    ss1 = OF._execute(extract=False)
    # setup spike sorting dict
    del OF
    SS = {'ss0':ss0, 'ss1':ss1}

    # walk though ground truth and process on overlaps
    rval = []
    res = align_spike_trains(GT, SS, MAX_SHIFT, MAX_JITTER, MAX_OVERLAP)['table'][1:]
    for item in res:
        rval.append(item[2:])
    rval = N.asarray(rval)

    # save and return
    if save:
        save = osp.join(save, exp, '%s_%04d_perf.csv' % (exp, trial))
        N.savetxt(save, rval, fmt='%4d', delimiter=',')
    return rval


def block_performance_analysis(exp='19022008', block='A', save=False):
    """analyse spike sorting performance for a whole block"""

    print 'perf_ana(block): %s %s' % (exp, block)

    rval = []
    for trial in EXP_DICT[exp][block]:
        rval.append(
            performance_analysis(
                exp,
                int(trial.split('_')[1][:4]),
                save=save
            )
        )
    rval = N.sum(rval, axis=0)

    # save and return
    if save:
        save = osp.join(save, exp, '%s_%s_perf.csv' % (exp, block))
        N.savetxt(save, rval, fmt='%4d', delimiter=',')
    return rval


def exp_performance_analysis(exp='19022008', save=False):
    """analyse spike sorting performance for a whole experiment"""

    print 'perf_ana(exp): %s' % exp

    rval = []
    for block in EXP_DICT[exp]:
        rval.append(
           block_performance_analysis(
               exp,
               block=block,
               save=save
            )
       )
    rval = N.sum(rval, axis=0)

    # save and return
    if save:
        save = osp.join(save, exp, '%s_perf.csv' % exp)
        N.savetxt(save, rval, fmt='%4d', delimiter=',')
    return rval


def everything_performance_analysis(save=False):
    """analyse spike sorting performance for all Alle Data"""

    print 'perf_ana(everything)'

    rval = []
    for exp in EXP_DICT:
        rval.append(
            exp_performance_analysis(
                exp,
                save=save
            )
        )
    rval = N.sum(rval, axis=0)

    # save and return
    if save:
        save = osp.join(save, 'perf.csv')
        N.savetxt(save, rval, fmt='%4d', delimiter=',')
    return rval


##---MAIN

if __name__ == '__main__':

    exp = '19022008'
    block = 'A'
    save = PICPATH

#    perf = performance_analysis(exp, 0, save)
#    block_performance_analysis(exp, block, save)
#    exp_performance_analysis(exp, save)
    everything_performance_analysis(save)

    print
    print
    print 'FINISHED!'
