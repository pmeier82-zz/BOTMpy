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


##---FUNCTIONS

def delta_tau_analysis(exp='19022008', trial=0, plot=True):
    """produce a delta_tau analysis plot per trial"""

    print 'dt_ana(trial): %s %04d' % (exp, trial)

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
    DT = N.array([
        N.arange(-MAX_OVERLAP, MAX_OVERLAP + 1),
        N.zeros(2 * MAX_OVERLAP + 1),
        N.zeros(2 * MAX_OVERLAP + 1)
    ]).T
    DT_kernel = lambda x: int(x) + MAX_OVERLAP
    idx = 0
    for idx in xrange(len(GT['gt0'])):
        check = check_inclusion(GT['gt0'][idx], GT['gt1'], MAX_OVERLAP)
        if check is not None:
            delta_tau = check - GT['gt0'][idx]
            DT[DT_kernel(delta_tau), 1] += 1
            ovp = overlap_detection(GT['gt0'][idx], check, SS, MAX_JITTER + MAX_SHIFT)
            if ovp is not None:
                DT[DT_kernel(delta_tau), 2] += 1

    # plot and return
    if plot:
        if isinstance(plot, str):
            plot = osp.join(plot, exp, '%s_%04d_delta_tau.svg' % (exp, trial))
        plot_dt(DT, plot)
    return DT


def block_delta_tau_analysis(exp='19022008', block='A', plot=True):
    """analyse delta tau performance for a whole block"""

    print 'dt_ana(block): %s %s' % (exp, block)

    DT = N.array([
        N.arange(-MAX_OVERLAP, MAX_OVERLAP + 1),
        N.zeros(2 * MAX_OVERLAP + 1),
        N.zeros(2 * MAX_OVERLAP + 1)
    ]).T

    for trial in EXP_DICT[exp][block]:
        dta = delta_tau_analysis(exp, int(trial.split('_')[1][:4]), plot=plot)
        DT[:, 1:] += dta[:, 1:]

    # plot and return
    if plot:
        if isinstance(plot, str):
            plot = osp.join(plot, exp, '%s_%s_delta_tau.svg' % (exp, block))
        plot_dt(DT, plot)
    return DT


def exp_delta_tau_analysis(exp='19022008', plot=True):
    """analyse delta tau performance for a whole experiment"""

    print 'dt_ana(exp): %s' % exp

    DT = N.array([
        N.arange(-MAX_OVERLAP, MAX_OVERLAP + 1),
        N.zeros(2 * MAX_OVERLAP + 1),
        N.zeros(2 * MAX_OVERLAP + 1)
    ]).T

    for block in EXP_DICT[exp]:
        dta = block_delta_tau_analysis(exp, block=block, plot=plot)
        DT[:, 1:] += dta[:, 1:]

    # plot and return
    if plot:
        if isinstance(plot, str):
            plot = osp.join(plot, exp, '%s_delta_tau.svg' % exp)
        plot_dt(DT, plot)
    return DT


def everything_delta_tau_analysis(plot=True):
    """analyse delta tau performance for all Alle Data"""

    print 'dt_ana(everything)'

    DT = N.array([
        N.arange(-MAX_OVERLAP, MAX_OVERLAP + 1),
        N.zeros(2 * MAX_OVERLAP + 1),
        N.zeros(2 * MAX_OVERLAP + 1)
    ]).T

    for exp in EXP_DICT:
        dta = exp_delta_tau_analysis(exp, plot=plot)
        DT[:, 1:] += dta[:, 1:]

    # plot and return
    if plot:
        if isinstance(plot, str):
            plot = osp.join(plot, 'delta_tau.svg')
        plot_dt(DT, plot)
    return DT


def overlap_detection(unit0, unit1, sorting, jitter=2):
    """check if overlap for unit0 and unit1 was detected in sorting with jitter"""
    u0 = check_inclusion(unit0, sorting['ss0'], jitter)
    u1 = check_inclusion(unit1, sorting['ss1'], jitter)
    if u0 is None or u1 is None:
        return None
    else:
        return True


def check_inclusion(sample, train, jitter=2):
    """check if sample is included in train with jitter"""
    for check in xrange(sample - jitter, sample + jitter + 1):
        if check in train:
            return check
    else:
        return None


def plot_dt(dtdata, plot_toggle):
    from plot import P
    fig = P.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_title('overlap sorting performance w.r.t. $\\Delta\\tau$')
    ax1.set_xlabel('$\\Delta\\tau$ in samples')
    ax1.set_ylabel('Performance in % correct')
    ax1.grid(True)
    perfo_line = ax1.plot(
        dtdata[:, 0],
        100.0 * dtdata[:, 2] / dtdata[:, 1],
        marker='x',
        color='r'
    )
    ax2 = P.twinx(ax1)
    count_line = ax2.plot(
        dtdata[:, 0],
        dtdata[:, 1],
        marker='x',
        color='b'
    )
    ax2.yaxis.tick_right()
    ax2.set_ylabel('occurrence count')
    ax1.set_ybound((-14, 105))
    ax2.set_ybound((dtdata[:, 1].max() * -0.2, dtdata[:, 1].max() * 1.5))
    ax1.legend(
        (perfo_line, count_line),
        ('performance', 'occurrences'),
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=2
    )

    if plot_toggle is True:
        P.show()
    else:
        fig.savefig(plot_toggle)
    P.close(fig)
    del fig


##---MAIN

if __name__ == '__main__':

    exp = '24022009'
    block = 'A'
    plot = True#PICPATH

#    res = delta_tau_analysis(exp, 0, plot)
#    res = block_delta_tau_analysis(exp, block, plot)
#    res = exp_delta_tau_analysis(exp, plot)
    res = everything_delta_tau_analysis(plot)
#    plot_dt(res, True)

    print
    print
    print 'FINISHED!'
