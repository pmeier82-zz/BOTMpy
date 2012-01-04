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


from multiprocessing import Process
import scipy as N
from os import path as osp
from tables import openFile
import logging
from util import *


##---CLASSES

class AllePlotting(Process):

    ## constructor

    def __init__(
        self,
        path=ATFPATH,
        exp='',
        block='A',
        dtype=N.float32,
        tf=67,
        delta=1.2
    ):
        """
        :Parameters:
            path : str
                path to the datafiles
            exp : str
                experiment identifier
            block : str
                block identifier in the experiment
            delta : int
                separation delta for overlaps
        """

        # checks
        if exp not in EXP_DICT:
            raise ValueError('exp not known')
        if block not in EXP_DICT[exp]:
            raise ValueError('block not known')

        # super
        super(AllePlotting, self).__init__(name='AllePlotting(%s-%s)' % (exp, block))

        # members
        self.exp = exp
        self.block = block
        self.arc = None
        self.tf = None

        # logger
        self.log = logging.getLogger('plotting_%s_%s' % (self.exp, self.block))
        self.log.setLevel(logging.INFO)
        self.log.addHandler(
            logging.FileHandler(
                osp.join(PICPATH, '%s-%s.plotting.log' % (self.exp, self.block)),
                mode='w'
            )
        )

        self.log.dbconfig('setup for %s-%s', self.exp, self.block)
        self.log.dbconfig('reading from %s, saving to %s', ATFPATH, HDFPATH)

    ## methods process

    def run(self):
        """process content"""

        # open archive
        self.log.dbconfig('opening archive')
        self.arc = openFile(
            osp.join(HDFPATH, '%s-%s-intermediate.h5' % (self.exp, self.block)),
            mode='r'
        )
        tf = None
        try:
            tf = N.asarray(self.arc.getNode('/%s/sweep00/sp_unit0' % EXP_DICT[self.exp][self.block][0]).read())
            assert tf.ndim == 2
        except:
            tf = N.asarray(self.arc.getNode('/%s/sweep00/sp_unit1' % EXP_DICT[self.exp][self.block][0]).read())
            assert tf.ndim == 2
        print tf
        tf = N.asarray(tf)
        if tf.shape[1] % 4 > 0:
            raise ValueError('waveform shape is not good for 4 channels!')
        self.tf = int(tf.shape[1] / 4)

        # generate plots
        self.log.dbconfig('starting to generate plots')
        for plot_func in self.__class__.__dict__:
            if plot_func.startswith('plot_'):
                getattr(self, plot_func)()

        self.arc.close()
        self.log.dbconfig('DONE: %s block %s' % (self.exp, self.block))

    def _plot_3d_blocks(self):

        # imports
        from mpl_toolkits.mplot3d import Axes3D
        from plot import P

        self.log.dbconfig('starting: 3d blocks')
        the_matrix = []
        for trial in EXP_DICT[self.exp][self.block]:
            for sweep in xrange(100):
                the_matrix.append(self.arc.getNode('/%s/sweep00/sp_unit0' % trial).read())
        the_matrix = N.vstack(the_matrix)
        x = N.arange(self.tf * 4)
        y = N.arange(the_matrix.shape[0])
        X, Y = N.meshgrid(x, y)

        self.log.dbconfig('plotting data')
        fig = P.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, the_matrix, cmap=P.cm.get_cmap('prism'))
        fig.suptitle('3d surface plot of spikes in %s block %s' % (self.exp, self.block))
        fig.savefig(osp.join(PICPATH, self.exp, '%s_block%s_3d.svg' % (self.exp, self.block)))
        P.close(fig)
        del fig, the_matrix, X, Y, x, y

        self.log.dbconfig('leaving: 3d blocks')

    def plot_amplitude_histogram(self):

        # imports
        from plot import P

        self.log.dbconfig('starting: amplitude progression')
        self.log.dbconfig('producing data')

        fig_block = P.figure()
        ax_block = []
        ax_block_cnt = 0
        for i in xrange(4):
            ax_block.append(fig_block.add_subplot(2, 2, i + 1))
            ax_block[i].set_xlabel('number of spikes')
            ax_block[i].set_ylabel(u'amplitude (�V)')
            ax_block[i].set_title('channel %d' % (i + 1))
        fig_block.suptitle('amplitude progression in %s block %s' % (self.exp, self.block))

        for trial in EXP_DICT[self.exp][self.block]:

            unit0_1st = []
            unit0_2nd = []
            unit1_1st = []
            unit1_2nd = []
            delta_tau = []
            for sweep in xrange(100):
                # delta tau
                time_u0 = self.arc.getNode('/%s/sweep%02d/gt_unit0' % (trial, sweep)).read()
                time_u1 = self.arc.getNode('/%s/sweep%02d/gt_unit1' % (trial, sweep)).read()
                dt = []
                for i in xrange(3):
                    if len(time_u0) > i and len(time_u1) > i:
                        if time_u0[i] == VOID or time_u1[i] == VOID:
                            dt.append(None)
                        else:
                            dt.append(int(time_u1[i]) - int(time_u0[i]))
                    else:
                        dt.append(None)
                delta_tau.append(dt)
                # cut spikes
                unit0 = self.arc.getNode('/%s/sweep%02d/sp_unit0' % (trial, sweep)).read()
                unit1 = self.arc.getNode('/%s/sweep%02d/sp_unit1' % (trial, sweep)).read()
                if isinstance(unit0, N.ndarray):
                    temp_u0_1st = []
                    temp_u0_2nd = []
                    for i in range(4):
                        if unit0.shape[0] > 0:
                            temp_u0_1st.append([
                                unit0[0, i * self.tf:(i + 1) * self.tf].min(),
                                unit0[0, i * self.tf:(i + 1) * self.tf].max()
                            ])
                        else:
                            temp_u0_1st.append([None, None])
                        if unit0.shape[0] > 1:
                            temp_u0_2nd.append([
                                unit0[1, i * self.tf:(i + 1) * self.tf].min(),
                                unit0[1, i * self.tf:(i + 1) * self.tf].max()
                            ])
                        else:
                            temp_u0_2nd.append([None, None])
                    unit0_1st.append(temp_u0_1st)
                    unit0_2nd.append(temp_u0_2nd)
                else:
                    unit0_1st.append([[None, None] for i in xrange(4)])
                    unit0_2nd.append([[None, None] for i in xrange(4)])
                if isinstance(unit1, N.ndarray):
                    temp_u1_1st = []
                    temp_u1_2nd = []
                    for i in range(4):
                        if unit1.shape[0] > 0:
                            temp_u1_1st.append([
                                unit1[0, i * self.tf:(i + 1) * self.tf].min(),
                                unit1[0, i * self.tf:(i + 1) * self.tf].max()
                            ])
                        else:
                            temp_u1_1st.append([None, None])
                        if unit1.shape[0] > 1:
                            temp_u1_2nd.append([
                                unit1[1, i * self.tf:(i + 1) * self.tf].min(),
                                unit1[1, i * self.tf:(i + 1) * self.tf].max()
                            ])
                        else:
                            temp_u1_2nd.append([None, None])
                    unit1_1st.append(temp_u1_1st)
                    unit1_2nd.append(temp_u1_2nd)
                else:
                    unit1_1st.append([[None, None] for i in xrange(4)])
                    unit1_2nd.append([[None, None] for i in xrange(4)])
            unit0_1st = N.asarray(unit0_1st)
            unit0_2nd = N.asarray(unit0_2nd)
            unit1_1st = N.asarray(unit1_1st)
            unit1_2nd = N.asarray(unit1_2nd)
            delta_tau = N.asarray(delta_tau)

            self.log.dbconfig('plotting data for %s' % trial)
            fig = P.figure()
            x_values = N.arange(ax_block_cnt, ax_block_cnt + 100)
            for i in xrange(4):
                ax = fig.add_subplot(2, 2, i + 1)
                ax.set_xlabel('number of spikes')
                ax.set_ylabel(u'amplitude (�V)')
                ax.set_title('channel %d' % (i + 1))
                ax.plot(unit0_1st[:, i, 0], 'b-')
                ax.plot(unit0_1st[:, i, 1], 'b-')
                ax.plot(unit1_1st[:, i, 0], 'r-')
                ax.plot(unit1_1st[:, i, 1], 'r-')
                ax.plot(delta_tau[:, 0], 'k')
#                ax.plot(unit0_2nd[:, i, 0], 'c-')
#                ax.plot(unit0_2nd[:, i, 1], 'c-')
#                ax.plot(unit1_2nd[:, i, 0], 'm-')
#                ax.plot(unit1_2nd[:, i, 1], 'm-')
                # add to block plot
                ax_block[i].axvline(ax_block_cnt, c='k')
                ax_block[i].plot(x_values, unit0_1st[:, i, 0], 'b-')
                ax_block[i].plot(x_values, unit0_1st[:, i, 1], 'b-')
                ax_block[i].plot(x_values, unit1_1st[:, i, 0], 'r-')
                ax_block[i].plot(x_values, unit1_1st[:, i, 1], 'r-')
                ax_block[i].plot(x_values, delta_tau[:, 0], 'k')
#                ax_block[i].plot(x_values, unit0_2nd[:, i, 0], 'c-')
#                ax_block[i].plot(x_values, unit0_2nd[:, i, 1], 'c-')
#                ax_block[i].plot(x_values, unit1_2nd[:, i, 0], 'm-')
#                ax_block[i].plot(x_values, unit1_2nd[:, i, 1], 'm-')
            ax_block_cnt += 100
            fig.suptitle('amplitude progression in %s block %s trial %s' % (self.exp, self.block, trial))
            fig.savefig(osp.join(PICPATH, self.exp, '%s_ampl_hist.svg' % trial), format='svg')
            P.close(fig)
            del fig

        for i in xrange(4):
            ax_block[i].axvline(ax_block_cnt, c='k')
        fig_block.savefig(osp.join(PICPATH, self.exp, '%s_%s_ampl_hist.svg' % (self.exp, self.block)), format='svg')
        P.close(fig_block)
        del fig_block

        self.log.dbconfig('leaving: amplitude blocks')


##---MAIN

if __name__ == '__main__':
    print "Starting..."
    import sys

#    proc_list = []
#    for exp in EXP_DICT:
#        for block in EXP_DICT[exp]:
#            proc_list.append(AllePlotting(exp=exp, block=block))
#    for proc in proc_list:
#        proc.start()
#    for proc in proc_list:
#        proc.join()

    proc = AllePlotting(exp='HA25022009', block='D')
    proc.run()

    print
    print
    print 'FINISHED!'
    sys.exit(0)
