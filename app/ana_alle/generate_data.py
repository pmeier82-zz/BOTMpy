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
import sys
# SpikePy packages
from common import TimeSeriesCovE, extract_spikes
from common.datafile import AtfFile
from nodes import SDIntraNode
# project packages
from util import *


##---CLASSES

class AlleDataGenerator(Process):

    ## constructor

    def __init__(
        self,
        path=ATFPATH,
        exp='',
        block='A',
        dtype=N.float32,
        tf=67
    ):
        """
        :Parameters:
            path : str
                path to the datafiles
            exp : str
                experiment identifier
            block : str
                block identifier in the experiment
            dtype : numpy.dtype
                dtype of the buffers
            tf : int
                template lenght
        """

        # checks
        if exp not in EXP_DICT:
            raise ValueError('exp not known')
        if block not in EXP_DICT[exp]:
            raise ValueError('block not known')

        # super
        super(AlleDataGenerator, self).__init__(name='AlleDataGenerator(%s-%s)' % (exp, block))

        # setting members
        self.path = path
        self.exp = exp
        self.block = block
        self.dtype = N.dtype(dtype)
        self.tf = tf

        # logger
        self.log = logging.getLogger('generate_data_%s_%s' % (self.exp, self.block))
        self.log.setLevel(logging.INFO)
        self.log.addHandler(
            logging.FileHandler(
                osp.join(HDFPATH, '%s-%s.intermediate.log' % (self.exp, self.block)),
                mode='w'
            )
        )

        self.log.dbconfig('setup for %s-%s', self.exp, self.block)
        self.log.dbconfig('reading from %s, saving to %s', ATFPATH, HDFPATH)

    ## methods process

    def run(self):
        """process content"""

        data_buf = {}
        gtruth = {}
        spikes = {}
        self.log.dbconfig('starting init')
        det_gt = SDIntraNode(tf=self.tf)
        Nest = TimeSeriesCovE(tf=self.tf)
        Rest = TimeSeriesCovE(tf=self.tf)
        ini_temps_read = {}
        ini_temps_mean = {}
        ini_temps = {}
        cut = (self.tf - 1) / 2

        self.log.dbconfig('generating save archive')
        save_file = openFile(
            osp.join(HDFPATH, '%s-%s-intermediate.h5' % (self.exp, self.block)),
            mode='w',
            title='Alle Data Storage [%s, block %s]' % (self.exp, self.block)
        )

        self.log.dbconfig('starting per file loop')
        for trial in EXP_DICT[self.exp][self.block]:

            self.log.dbconfig('reading %s', trial)
            grp_trial = save_file.createGroup(save_file.root, '%s' % trial)

            read_file = AtfFile(osp.join(self.path, self.exp, trial), dtype=self.dtype)
            data_buf[trial] = {}
            gtruth[trial] = {}
            spikes[trial] = {}

            # for each sweep in the file
            for sweep in range(100):

                self.log.debug('exp %s block %s sweep %d' % (self.exp, self.block, sweep))
                grp_sweep = save_file.createGroup(grp_trial, 'sweep%02d' % sweep)

                # data_buf the data
                data_buf[trial][sweep] = read_file.get_data(mode='data', item=sweep).T
                data_buf[trial][sweep][:, 2:] -= data_buf[trial][sweep][:, 2:].mean(0)
                save_file.createArray(grp_sweep, 'data', data_buf[trial][sweep])

                # ground truth
                gtruth[trial][sweep] = {}
                gtruth[trial][sweep][0] = det_gt(data_buf[trial][sweep][:, 0:1])
                if len(gtruth[trial][sweep][0]) == 0:
                    gtruth[trial][sweep][0] = [VOID]
                save_file.createArray(grp_sweep, 'gt_unit0', gtruth[trial][sweep][0])
                gtruth[trial][sweep][1] = det_gt(data_buf[trial][sweep][:, 1:2])
                if len(gtruth[trial][sweep][1]) == 0:
                    gtruth[trial][sweep][1] = [VOID]
                save_file.createArray(grp_sweep, 'gt_unit1', gtruth[trial][sweep][1])

                # noise estimation
                Nest = TimeSeriesCovE(tf=self.tf)
                Nest.update(data_buf[trial][sweep][5000:, 2:])
                save_file.createArray(grp_sweep, 'CovNoise', Nest.get_covmx())
                Rest = TimeSeriesCovE(tf=self.tf)
                Rest.update(data_buf[trial][sweep][:, 2:])
                save_file.createArray(grp_sweep, 'CovData', Rest.get_covmx())

                # cut out spikes
                spikes[trial][sweep] = {}
                spikes[trial][sweep][0] = []
                for spk_t in gtruth[trial][sweep][0]:
                    if spk_t != VOID:
                        spikes[trial][sweep][0].append(
                            extract_spikes(
                                data_buf[trial][sweep][:, 2:],
                                N.array([[spk_t - cut, spk_t + cut + 1]])
                            )[0]
                        )
                if len(spikes[trial][sweep][0]) == 0:
                    spikes[trial][sweep][0] = [VOID]
                else:
                    spikes[trial][sweep][0] = N.vstack(spikes[trial][sweep][0])
                save_file.createArray(grp_sweep, 'sp_unit0', spikes[trial][sweep][0])
                spikes[trial][sweep][1] = []
                for spk_t in gtruth[trial][sweep][1]:
                    if spk_t != VOID:
                        spikes[trial][sweep][1].append(
                            extract_spikes(
                                data_buf[trial][sweep][:, 2:],
                                N.array([[spk_t - cut, spk_t + cut + 1]])
                            )[0]
                        )
                if len(spikes[trial][sweep][1]) == 0:
                    spikes[trial][sweep][1] = [VOID]
                else:
                    spikes[trial][sweep][1] = N.vstack(spikes[trial][sweep][1])
                save_file.createArray(grp_sweep, 'sp_unit1', spikes[trial][sweep][1])

            # close archive
            read_file.close()
            del read_file

        save_file.close()
        data_buf.clear()
        gtruth.clear()
        spikes.clear()

        self.log.dbconfig('DONE: %s block %s' % (self.exp, self.block))


##---MAIN

if __name__ == '__main__':

    proc_list = []
    for exp in EXP_DICT:
        for block in EXP_DICT[exp]:
            proc_list.append(AlleDataGenerator(exp=exp, block=block))
    for proc in proc_list:
        proc.start()
    for proc in proc_list:
        proc.join()

#    proc = [AlleDataGenerator(exp='HA25022009', block='A'),
#            AlleDataGenerator(exp='HA25022009', block='B')]
#    for p in proc:
#        p.start()
#    for p in proc:
#        p.join()

    print
    print
    print 'FINISHED!'
    sys.exit(0)
