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


"""NTrode handlers for the Munk data analysis for establishing a multi-unit"""
__docformat__ = 'restructuredtext'


##--- IMPORTS

import scipy as sp
from ntrode import NTrodeHandler, NTrodeError
from nodes import ArtifactDetectorNode, SDMteoNode as SDCls
from database import MunkSession, CON_GNODE
from tetrode import Tetrode


##---CONSTANTS

KV = [3 + i * 5 for i in xrange(6)]


##---CLASSES

class MunkMUDBInputHandler(NTrodeHandler):
    """input handler for munk data from the munk database"""

    def __init__(self, db=None, mean_correct=False):
        """
        :Parameters:
            db : MunkPostgresDB
                database handle
            mean_correct : bool
                If True, mean correct the data before processing it.
        """

        # super
        super(MunkMUDBInputHandler, self).__init__()

        # members
        self.db = db
        self.mean_correct = bool(mean_correct)
        self.trial_ids = None
        self.idx = None
        self.art_det = None
        self.id_exp = None
        self.id_blk = None
        self.id_tet = None

        # method mapping
        self.invoke_for_state.update(INPUT=self._on_input)

    def _initialise(self):
        """called during INIT state"""

        # get database ids
        self.idx = -1
        self.art_det = ArtifactDetectorNode()
        self.id_exp = self.db.get_exp_id(self.mem['exp'])
        self.id_blk = self.db.get_block_id(self.mem['exp'], self.mem['blk'])
        self.id_tet = self.db.get_tetrode_id(self.id_exp, self.mem['tet'])

        # register memory
        self.mem.update(
            rbuf=None,
            input_idx='MunkMUDBInputHandler._initialise',
            art_det=self.art_det,
            art_ep=None,
            art_good=True
        )

        # build lists
        self.trial_ids = self.db.get_trial_range(self.id_blk, include_error_trials=False)
        if len(self.trial_ids) == 0:
            raise NTrodeError('empty trial list')

    def _on_input(self):
        """called during INPUT state"""

        # reset rbuf
        self.mem.update(rbuf=None)

        # check for end of sequence
        if self.idx + 1 != len(self.trial_ids):
            self.idx += 1
        else:
            self.mem.update(input_idx=None)
            return

        # read data
        data = None
        try:
            data = self.db.get_tetrode_data(self.trial_ids[self.idx],
                self.id_tet,
                sp.dtype(self.mem['dtype']))
            self.mem.update(input_idx=self.trial_ids[self.idx])
            print 'read %s' % self.db.get_fname_for_id(self.mem['input_idx'])
        except Exception, e:
            print 'error reading input data for: %s' % self.db.get_fname_for_id(self.trial_ids[self.idx])
            print e
            self.mem.update(input_idx=None)
            return

        # check data
        if self.mean_correct is True:
            data -= data.mean(axis=0)
        self.mem.update(rbuf=data)

        # artifact detection
        self.art_det(data)
        self.mem['art_ep'] = self.art_det.events
        self.mem['good_ep'] = self.art_det.get_nonartefact_epochs()
        art_crit = self.art_det.get_fragmentation()
        print 'artifact fragmentation: %s' % art_crit
        if art_crit > 1:
            # TODO: find reasonable ratio and fragmentation degree
            self.mem.update(art_good=False)
            return
        else:
            self.mem.update(art_good=True)
        self.mem.update(rbuf=data)


class MunkMUDetectionHandler(NTrodeHandler):
    """handler to applying the spikesorting to the data"""

    def __init__(self, th_fac=None):
        """
        :Parameters:
            None
        """

        # super
        super(MunkMUDetectionHandler, self).__init__()

        # members
        self.th_fac = th_fac or 3.5
        self.sdet = None

        # method mapping
        self.invoke_for_state.update(PROCESS=self._on_process)

    def _initialise(self):
        """called during INIT state"""

        # setup spike detector and noise estimator
        self.sdet = SDCls(kvalues=KV,
            threshold_factor=self.th_fac,
            tf=self.mem['tf'],
            min_dist=int(self.mem['tf'] * 0.5))

    def _on_process(self):
        """perform detection"""

        # exit check
        if self.mem['input_idx'] is None or self.mem['art_good'] is False:
            return
        self.mem['mu'] = sp.zeros((0, 2))

        # process all epochs for this trial
        self.sdet(self.mem['rbuf'])
        mu = self.sdet.events
        for ep in self.mem['good_ep']:
            mu = mu[(mu >= ep[0]) * (mu <= ep[1])]

        # save sorting to mem
        self.mem.update(mu=mu)


class MunkMUDBOutputHandler(NTrodeHandler):
    """database output handler"""

    def __init__(self, db=None):
        """
        :Parameters:
            db : MunkPostgresDB
        """

        # super
        super(MunkMUDBOutputHandler, self).__init__()

        # members
        self.db = db
        self.id_exp = None
        self.id_blk = None
        self.id_tet = None
        self.id_ana = None
        self.id_unt = None

        # break vars
        self.art_bad_cnt = 0

        # method mapping
        self.invoke_for_state.update(OUTPUT=self._on_output)

    def _initialise(self):
        """called during INIT state"""

        # setup analysis
        self.id_exp = self.db.get_exp_id(self.mem['exp'])
        self.id_blk = self.db.get_block_id(self.mem['exp'],
            self.mem['blk'])
        self.id_tet = self.db.get_tetrode_id(self.id_exp,
            self.mem['tet'])
        trial_start = self.db.get_trial_range(self.id_blk, limit=1)[0]
        q = self.db.query("""
        INSERT INTO analysis (expid, block, tetrode, trialidxstart, trialidxend,
                              algorithm, kind)
        VALUES (%d, %d, %d,
                (SELECT t.trialidx FROM trial t WHERE t.id = %d),
                (SELECT t.trialidx FROM trial t WHERE t.id = %d),
                '%s', '%s')
        RETURNING currval('analysis_id_seq');
        """ % (self.id_exp, self.id_blk, self.id_tet, trial_start, trial_start,
               self.mem['algo'], 'MUA'),
            commit=True)
        self.id_ana = q[0][0]
        self.id_unt = self.db.insert_unit(0, self.id_ana, 'multi-unit')

    def _on_output(self):
        """output hook"""

        # exit check
        if self.mem['input_idx'] is None:
            return
        if self.mem['art_good'] is False:
            self.art_bad_cnt += 1
            if self.art_bad_cnt == 5:
                self.db.query("""
                UPDATE analysis a
                SET status = 'aborted due to artifact fragmentation'
                WHERE a.id = %d
                """ % self.id_ana,
                    commit=True)
                self.mem.update(input_idx=None)
            return
        else:
            self.art_bad_cnt = 0

            # for all units write spikes into db
        #        if self.mem['art_ep'].size > 0:
        #            self.insert_art_epochs(self.mem['art_ep'])
        # need to improve artifact epoch handling first:/
        if self.mem['mu'].size > 0:
            self.db.insert_spiketrain(self.mem['input_idx'], self.id_unt, self.mem['mu'])

        # update the trialidxend in analysis
        self.update_trialidxend()

    def update_trialidxend(self):
        """update the trialidxend after a trial is finished"""

        self.db.query("""
        UPDATE analysis a
        SET trialidxend = (SELECT t.trialidx FROM trial t WHERE t.id = %d)
        WHERE a.id = %d
        """ % (self.mem['input_idx'], self.id_ana),
            commit=True)

    def insert_art_epochs(self, epochs):
        """insert artifact epochs in to the artifact table"""

        # exit check
        if len(epochs) == 0:
            return

        self.db.query("""
        INSERT INTO artifact (trial, tetrode, sample_start, sample_end)
        VALUES %s
        """ % ',\n'.join(map(str, [(self.mem['input_idx'], self.id_tet, ep[0], ep[1])
        for ep in epochs])),
            commit=True)


##--- MAIN

if __name__ == '__main__':
    # imports
    import sys, traceback

    # inits
    if len(sys.argv) != 4:
        sys.exit('give exactly 3 arguments: 1) exp, 2) block, 3) tetrode !!!')
    exp = str(sys.argv[1])
    blk = str(sys.argv[2]).lower()
    tet = int(sys.argv[3])
    # db session
    DB_INFO = CON_GNODE
    DB = MunkSession(dbconfig=DB_INFO)
    DB.connect()
    # check experiment
    id_exp = DB.get_exp_id(exp)
    if id_exp is None:
        sys.exit('no matching experiment for: %s' % exp)
        # check block
    id_blk = DB.get_block_id(exp, blk)
    if id_blk == None:
        sys.exit('no matching block for: %s-%s' % (exp, blk))
        # check tetrode
    id_tet = DB.get_tetrode_id(id_exp, tet)
    print id_tet, tet, '__main__'
    if id_tet is None:
        sys.exit('no matching tetrode for: %s-%s-%s' % (exp, blk, tet))

    # start the tetrode
    try:
        print
        print '########################'
        print 'starting Tetrode'
        print 'exp:', exp, 'blk:', blk, 'tet:', tet
        print '########################'
        print

        Tetrode(
            name='MU: %s %s %s' % (exp, blk, tet),
            exp=exp,
            blk=blk,
            tet=tet,
            algo='mTeo(%s)' % KV,
            # internals
            tf=65,
            nc=4,
            dtype=sp.float32,
            # handlers
            handlers=[
                (MunkMUDBInputHandler, {'db': DB}),
                (MunkMUDetectionHandler, {}),
                (MunkMUDBOutputHandler, {'db': DB})
            ]
        ).run()

    except Exception, ex:
        print '##########################'
        print 'ERROR while processing:', exp, blk, tet
        traceback.print_exception(*sys.exc_info())
        print '##########################'
        print

    finally:
        DB.close()
        del DB

    print
    print 'ALL DONE'
    print
