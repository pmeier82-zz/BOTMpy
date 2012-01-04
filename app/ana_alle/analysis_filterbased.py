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


from multiprocessing import Process, log_to_stderr
import scipy as sp
from os import path as osp
from tables import openFile
import logging
import sys
from common import (TimeSeriesCovE, extract_spikes, INDEX_DTYPE, AtfFile,
                    GdfFile, epochs_from_spiketrain_set, mcvec_from_conc,
                    get_cut)
from nodes import FilterBankSortingNode, FSSNode
from util import *


##---CLASSES

class AlleAnalysis(Process):

    ## constructor

    def __init__(
        self,
        atf_path,
        gdf_path,
        exp,
        block,
        tf=67,
        cut_offset=0,
        delta=30,
        ss_cls=None,
        dtype=sp.float32
    ):
        """
        :Parameters:
            atf_path : str
                path to the directory where the atf data files reside
            gdf_path : str
                path to the directory where the gdf groundtruth files reside
            exp : str
                experiment identifier
            block : str
                block identifier in the experiment
            tf : int
                template length
            cut_offset : int
                offsets the cut values to the right or left in samples
                Default=0
            delta : int
                separation delta for overlaps in samples (should be tf/2 or less)
            ss_cls : SortingNode
                class to use for the spike sorting
                Default=FSSNode
            dtype : numpy.dtype
                dtype of the buffers
        """

        # checks
        if exp not in EXP_DICT:
            raise ValueError('exp not known')
        if block not in EXP_DICT[exp]:
            raise ValueError('block not known')
        if not issubclass(ss_cls, FilterBankSortingNode):
            raise TypeError('ss_cls has to be a subclass of SortingNode')

        # super
        super(AlleAnalysis, self).__init__(name='AlleAnalysis(%s-%s)' % (exp, block))

        # external members
        self.atf_path = atf_path
        self.gdf_path = gdf_path
        self.exp = exp
        self.block = block
        self.dtype = sp.dtype(dtype)
        self.tf = int(tf)
        self.cut_offset = int(cut_offset)
        self.delta = int(delta)
        self.ss_cls = ss_cls

        # internal members
        self.buffer = {}
        self.gtruth = {}
        self.sorted = {}
        self.cov_est = None
        self.ini_temps = None
        self.ini_temps_buf = None
        self.history = {}

        # logger
        self.log = log_to_stderr()
        self.log.setLevel(logging.DEBUG)
        self.log.addHandler(
            logging.FileHandler(
                osp.join(HDFPATH, '%s-%s.log' % (self.exp, self.block)),
                mode='w'
            )
        )

        self.log.dbconfig('setup for %s-%s', self.exp, self.block)
        self.log.dbconfig('reading from %s, saving to %s', ATFPATH, HDFPATH)

    ## methods process

    def run(self):
        """process content"""

        self.log.dbconfig('starting in process')

        self.ana_init()
        self.ana_run()
        self.ana_postprocessing()

        self.log.dbconfig('finishing in process')

    ## methods analysis

    def ana_init(self):
        """initialise for analysis"""

        ## setup internals

        self.log.debug('starting initialisation')

        self.cov_est = TimeSeriesCovE(tf=self.tf, nc=4)
        self.ini_temps_buf = {}
        self.ini_temps = {}
        cut = get_cut(self.tf, off=self.cut_offset)

        ## per file init

        self.log.dbconfig('starting init loop')

        # for each trial in the block
        for trial in EXP_DICT[self.exp][self.block]:

            self.log.dbconfig('reading %s', trial)

            arc = AtfFile(osp.join(self.atf_path, self.exp, trial), dtype=self.dtype)
            self.buffer[trial] = arc.get_data(mode='device', item=1)
            self.buffer[trial] -= self.buffer[trial].mean(0)
            self.gtruth[trial] = GdfFile.read_gdf(osp.join(self.gdf_path,
                                                           ''.join([trial[:-4], '.gdf'])))

            # matlab indexing fix! - START
            for u in self.gtruth[trial]:
                self.gtruth[trial][u] -= 1
            # matlab indexing fix! - END

            # epoch sets
            epoch_set = epochs_from_spiketrain_set(self.gtruth[trial], cut, end=1000000)
            ep_u0 = []
            for i in xrange(len(epoch_set['00001'])):
                if self.gtruth[trial]['00002'][(self.gtruth[trial]['00002'] < epoch_set['00001'][i][0] - self.delta) *
                                               (self.gtruth[trial]['00002'] > epoch_set['00001'][i][1] + self.delta + 1)].size > 0:
                    continue
                else:
                    ep_u0.append(i)
            ep_u1 = []
            for i in xrange(len(epoch_set['00002'])):
                if self.gtruth[trial]['00001'][(self.gtruth[trial]['00001'] < epoch_set['00002'][i][0] - self.delta) *
                                               (self.gtruth[trial]['00001'] > epoch_set['00002'][i][1] + self.delta + 1)].size > 0:
                    continue
                else:
                    ep_u1.append(i)

            # extract spikes for units
            self.ini_temps_buf[trial] = {}
            self.ini_temps_buf[trial][0] = extract_spikes(self.buffer[trial],
                                                          epoch_set['00001'][ep_u0])
            self.ini_temps_buf[trial][1] = extract_spikes(self.buffer[trial],
                                                          epoch_set['00002'][ep_u1])

            # update noise covariance estimator
            self.cov_est.update(self.buffer[trial], epochs=epoch_set['noise'])

            # close archive
            arc.close()

        ## finalise inits

        self.log.dbconfig('building initial templates')

        self.ini_temps = sp.vstack([
            sp.vstack([self.ini_temps_buf[t][0] for t in self.ini_temps_buf.keys()]).mean(axis=0),
            sp.vstack([self.ini_temps_buf[t][1] for t in self.ini_temps_buf.keys()]).mean(axis=0)
        ])
        itemps = sp.zeros((2, self.tf, 4))
        itemps[0] = mcvec_from_conc(self.ini_temps[0], nc=4)
        itemps[1] = mcvec_from_conc(self.ini_temps[1], nc=4)
        self.sorter = self.ss_cls(
            itemps,
            ce=self.cov_est,
            det_th=0.5
        )

        self.log.dbconfig('leaving initialisation')

    def ana_run(self):
        """run analysis"""

        self.log.dbconfig('starting analysis')

        for trial in EXP_DICT[self.exp][self.block]:
            self.sorted[trial] = {}
            self.log.dbconfig('ana: %s', trial)
            self.sorter(self.buffer[trial])
            self.sorted[trial][0] = self.sorter.rval[0]
            self.sorted[trial][1] = self.sorter.rval[1]
            self.history[trial] = [self.sorter.template_set, self.sorter.ce.cmx]

        self.log.dbconfig('leaving analysis')

    def ana_postprocessing(self):
        """postprocess sorting"""

        self.log.dbconfig('starting postprocessing')

        # create archive
        arc_fname = osp.join(HDFPATH, '%s-%s.h5' % (self.exp, self.block))
        arc = openFile(
            arc_fname,
            mode='w',
            title='Alle Daten - Analyse[%s, block %s]' % (self.exp, self.block)
        )

        # init group
        grp_init = arc.createGroup(
            arc.root,
            'init',
            'initialisation information'
        )
        arc.createArray(grp_init, 'tf', self.tf)
        arc.createArray(grp_init, 'temps', self.ini_temps)
        arc.createArray(grp_init, 'cov', self.cov_est.cmx)

        # sortings
        for trial in EXP_DICT[self.exp][self.block]:

            # updated templates
            grp_trial = arc.createGroup(arc.root, trial)
            arc.createArray(
                grp_trial,
                'temps',
                self.history[trial][0]
            )
            # updates covariance matrix
            arc.createArray(
                grp_trial,
                'cov',
                self.history[trial][1]
            )
            # sorting unit 0
            ss_unit0 = self.sorted[trial][0]
            if ss_unit0.size == 0:
                ss_unit0 = sp.array([-1], dtype=INDEX_DTYPE)
            arc.createArray(grp_trial, 'ss_unit0', ss_unit0)
            # sorted unit1
            ss_unit1 = self.sorted[trial][1]
            if ss_unit1.size == 0:
                ss_unit1 = sp.array([-1], dtype=INDEX_DTYPE)
            arc.createArray(grp_trial, 'ss_unit1', ss_unit1)
            # flush archive
            arc.flush()

        arc.close()

        self.log.dbconfig('producing gdfs')

        gdf_from_hdf(arc_fname, HDFPATH)

        self.log.dbconfig('leaving postprocessing')

def gdf_from_hdf(hdf_path, gdf_path):
    """generates a gdf file from the sorting in the corresponding hdf file"""

    with openFile(hdf_path) as arc:
        for node in arc.root:
            if node._v_name[:4] in ['1902', '2402', 'HA25']:
                GdfFile.write_gdf(osp.join(gdf_path, ''.join([node._v_name, '.gdf'])),
                          dict(zip(xrange(1, 3), [node.ss_unit0.read(), node.ss_unit1.read()])))



##---MAIN

if __name__ == '__main__':

#    # build ALL
#    for exp in EXP_DICT:
#        if exp == 'test':
#            continue
#        print 'EXP:', exp
#        for block in EXP_DICT[exp]:
#            if block == 'test':
#                continue
#            print 'BLOCK:', block
#            ANA = AlleAnalysis(ATFPATH, GDFPATH, exp=exp, block=block)
#            ANA.run()

    # build SINGLE    
    ANA = AlleAnalysis(ATFPATH, GDFPATH, exp='19022008', block='A',
                       ss_cls=FSSNode, tf=95, cut_offset=20, delta=40)
    ANA.run()

    print 'FINISHED!'
    sys.exit(0)
