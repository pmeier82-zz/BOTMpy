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
from common import align_spike_trains
import os.path as osp
import scipy as N
from tables import openFile
from util import *


##---CONSTANTS

THRESHOLDS = N.arange(0.5, 0.91, 0.02)


##---FUNCTIONS

def roc_curve(exp='19022008', trial=0, plot=True):
    """produce a roc curve
    
    returns the ROC table |th|fp|tp|
    """

    # get data from archive
    arc = openFile(osp.join(HDFPATH, '%s_fss.h5' % exp), 'r')
    GT = {
        'gt0' : arc.getNode('/%s_%04d/gt_unit0' % (exp, trial)).read(),
        'gt1' : arc.getNode('/%s_%04d/gt_unit1' % (exp, trial)).read(),
    }
    fouts = arc.getNode('/%s_%04d/ss_fouts' % (exp, trial)).read()
    arc.close()
    del arc

    # roc data structure - varying
    ROC = N.zeros((THRESHOLDS.size, 3))
    for i in xrange(THRESHOLDS.size):

        OF = ThresholdDetectorNode(
            N.zeros(2),
            N.eye(2),
            N.eye(2),
            th_fac=THRESHOLDS[i]
        )
        # do for unit0
        OF.energy = N.atleast_2d(fouts[0].copy()).T
        OF.size, OF.nchan = OF.energy.shape
        ss0 = OF._execute(extract=False)

        # do for unit1
        OF.energy = N.atleast_2d(fouts[1].copy()).T
        OF.size, OF.nchan = OF.energy.shape
        ss1 = OF._execute(extract=False)
        del OF
        SS = {'ss0':ss0, 'ss1':ss1}

        # compare spiketrains
        T = align_spike_trains(GT, SS)['table']
        tp = sum(T[1][5:7] + T[2][5:7])
        fp = sum(T[1][9:11] + T[2][9:11] + [T[2][13], T[2][13]])
        fn = sum(T[1][11:13] + T[2][11:13])
        ROC[i] = THRESHOLDS[i], fp, tp

    # plot and return
    if plot:
        if isinstance(plot, str):
            plot = osp.join(PICPATH, exp, '%s_%s_roc.svg' % (exp, block))
        plot_roc(ROC, plot)
    return ROC


def block_roc(exp='19022008', block='A', plot=True):
    """genereates a roc curve for the given experiment and block
    
    returns the ROC table |th|fp|tp|
    """

    # checks
    assert exp in EXP_DICT, '%s i not in the EXP_DICT' % exp
    assert block in EXP_DICT[exp], 'no block %s in the EXP_DICT for exp %s' % (block, exp)
    print 'starting roc chart for exp:', exp, 'block:', block

    # cumulate roxx
    ROC = N.zeros((THRESHOLDS.size, 3))
    ROC[:, 0] = THRESHOLDS
    for trial in EXP_DICT[exp][block]:
        print 'trial:', trial
        ROC[:, 1:] += roc_curve(exp, int(trial.split('_')[1][:4]), plot=False)[:, 1:]

    # plot and return
    if plot:
        if isinstance(plot, str):
            plot = osp.join(PICPATH, exp, '%s_%s_roc.svg' % (exp, block))
        plot_roc(ROC, plot)
    return ROC


def plot_roc(rocdata, plot_toggle):
    from plot import P
    fig = P.figure()
    ax = fig.add_subplot(111)
    ax.plot(rocdata[::-1, 1], rocdata[::-1, 2])
    ax.set_xlabel('False Positives')
    ax.set_ylabel('True Positives')
    ax.set_title('ROC Chart - varying thresholds [%s - %s]' % (rocdata[0, 0], rocdata[-1, 0]))
    for item in rocdata:
        ax.annotate(
            'th:%s' % item[0],
            xy=(item[1], item[2]),
            xycoords='data',
            xytext=(20, -20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='->')
        )

    if plot_toggle is True:
        P.show()
    else:
        fig.savefig(plot_toggle)
    P.close(fig)


##---MAIN

if __name__ == '__main__':

    from sys import exit

    for exp in EXP_DICT:
        for block in EXP_DICT[exp]:
            block_roc(exp, block, plot=PICPATH)

    print
    print
    print 'FINISHED!'
    exit(0)
