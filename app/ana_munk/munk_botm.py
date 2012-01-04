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


"""NTrode handlers for the Munk data analysis applying BOTM sorting"""
__docformat__ = 'restructuredtext'


##--- IMPORTS

import scipy as sp
from ntrode import NTrodeHandler, NTrodeError
from tetrode import Tetrode
from nodes import BOTMNode
from common import TimeSeriesCovE, get_cut
import util


##--- CLASSES

class MunkDBInputHandler(NTrodeHandler):
    """input handler for trial data from the munk database"""

    def __init__(self, db=None, mean_correct=False, limit=None):
        """
        :Parameters:
            db : MunkPostgresDB
                database handle
            mean_correct : bool
                If True, mean correct the data before processing it.
            limit: int or None
                If not None limit the input trial range
        """

        # super
        super(MunkDBInputHandler, self).__init__()

        # members
        self.db = db
        self.mean_correct = bool(mean_correct)
        self.trial_ids = None
        self.idx = None
        self.id_exp = None
        self.id_tet = None
        self.limit = limit

        # method mapping
        self.invoke_for_state.update(INPUT=self._on_input)

    def _initialise(self):
        """called during INIT state"""

        self.mem.update(input_idx='MunkDBInputHandler._initialise')

        # get database ids
        self.idx = -1
        self.id_exp = self.db.get_exp_id(self.mem['exp'])
        self.id_tet = self.db.get_tetrode_id(self.id_exp, self.mem['tet'])

        # build lists
        self.trial_ids = []
        for blk in ['a', 'b', 'c', 'd', 'e']:
            id_blk = self.db.get_block_id(self.mem['exp'], blk)
            if id_blk is None:
                continue
            self.trial_ids.extend(
                self.db.get_trial_range(id_blk,
                                        include_error_trials=False,
                                        limit=self.limit))
        if len(self.trial_ids) == 0:
            raise NTrodeError('empty trial list')

        # register memory
        self.mem.update(rbuf=None,
                        id_exp=self.id_exp,
                        id_tet=self.id_tet,
                        trial_ids=self.trial_ids)

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
            print 'processing %s' % self.db.get_fname_for_id(
                self.mem['input_idx'])
        except Exception, e:
            print 'error processing %s' % self.db.get_fname_for_id(
                self.trial_ids[self.idx])
            print e
            self.mem.update(input_idx=None)
            return

        # check data
        if self.mean_correct is True:
            data -= data.mean(axis=0)
        self.mem.update(rbuf=data)


class MunkBOTMHandler(NTrodeHandler):
    """handler to applying the BOTM spikesorting to the data"""

    def __init__(self, chunk_size=100000, ovlp_tau=None, db=None):
        """
        :Parameters:
            None
        """

        # super
        super(MunkBOTMHandler, self).__init__()

        # members
        self.db = db
        self.ss = None
        self.ce = None

        self._chunk_size = int(chunk_size)
        self._ovlp_tau = ovlp_tau

        # method mapping
        self.invoke_for_state.update(PROCESS=self._on_process)

    def _initialise(self):
        """called during INIT state"""

        self.mem.update(input_idx='MunkBOTMHandler._initialise')

        print 'loading initialisation from database'
        print 'checking analysis..'
        q = self.db.query("""
        SELECT a.kind, a.expid, a.block, a.tetrode, t.id
        FROM analysis a
        JOIN trial t ON (t.block=a.block AND t.trialidx=a.trialidxstart)
        WHERE a.id = %d
        """ % self.mem['id_ini'])[0]
        if q[0] != 'INIT':
            raise NTrodeError(
                'Analysis(%d) is of type: %s; required: INIT!' % (
                self.mem['id_ini'], q[0]))
        if q[1] != self.mem['id_exp']:
            raise NTrodeError(
                'Initialisation(%d) is for another experiment!' % self.mem[
                                                                  'id_ini'])
        #        if q[2] != self.mem['id_blk']:
        #            raise NTrodeError('Initialisation(%d) is for another
        # block in the experiment!' % self.mem['id_ini'])
        if q[3] != self.mem['id_tet']:
            raise NTrodeError(
                'Initialisation(%d) is for another tetrode!' % self.mem[
                                                               'id_ini'])
        id_trl = int(q[4])
        print 'good.'

        print 'loading analysis data..'
        ulist = self.db.get_units_for_analysis(self.mem['id_ini'])
        if len(ulist) == 0:
            raise NTrodeError('did not find any units from initialisation!')
        units_ = []
        cl = 0
        for id_unt in ulist:
            udata = self.db.get_unit_data(id_unt, id_trl)
            if udata['waveform'] is None:
                raise NTrodeError('could not find waveform data from '
                                  'initialisation: id_unt:%d id_trl:%d'
                % (id_unt, id_trl))
            if udata['waveform']['data'] is None:
                raise NTrodeError('could not find waveform data from '
                                  'initialisation: id_unt:%d id_trl:%d'
                % (id_unt, id_trl))
            units_.append(udata['waveform']['data'])
            cl = udata['waveform']['cutleft']
        self.mem.update(cutleft=cl)
        if len(units_) == 0:
            raise NTrodeError('did not find any units from initialisation!')
        units = sp.zeros((len(units_), units_[0].shape[0],
                          units_[0].shape[1]))
        for i in xrange(len(units_)):
            units[i] = units_[i]
        del units_
        # get_covariance(db, id_ana, id_trl=None, kind=None)
        cmx_store = self.db.get_covariance(self.mem['id_ini'], id_trl,
                                           'noise')
        print 'good.'

        print 'building sorter and noise-estimator'
        self.mem.update(tf=units.shape[1],
                        nc=units.shape[2],
                        algo=self.mem['algo'] + '(tf=%d)' % units.shape[
                                                            1] + '(ini=%d)' %
                                                                 self.mem[
                                                                 'id_ini'])
        self.ce = TimeSeriesCovE(tf_max=self.mem['tf'],
                                 nc=self.mem['nc'])
        self.ce.new_chan_set((0, 1, 2, 3))
        self.ce._store = cmx_store
        self.ce._is_initialised = True
        ## sorting node
        #templates,
        #chan_set=(0, 1, 2, 3),
        #ce=None,
        #rb_cap=350,
        #adapt_templates=0,
        #learn_noise=True,
        #chunk_size=32000,
        #use_history=False,
        #debug=False,
        #dtype=None,
        ## bss node
        #ovlp_taus=[-2, 0, 2],
        #ovlp_meth='och',
        #spk_pr=1e-3,
        #noi_pr=1e0,
        self.ss = BOTMNode(templates=units,
                           chan_set=(0, 1, 2, 3),
                           ce=self.ce,
                           ovlp_taus=None,
                           adapt_templates=self.mem['align_at'],
                           learn_noise=True,
                           use_history=True,
                           debug=self.mem['debug'])
        self.mem.update(ss=self.ss, ce=self.ce)

    def _on_process(self):
        """perform sorting"""

        # exit check
        if self.mem['input_idx'] is None:
            return

        # chunked sorting
        self.ss(self.mem['rbuf'])
        self.mem.update(sorting=self.ss.rval)


class MunkDBOutputHandler(NTrodeHandler):
    """database output handler"""

    def __init__(self, db=None):
        """
        :Parameters:
            db : MunkPostgresDB
        """

        # super
        super(MunkDBOutputHandler, self).__init__()

        # members
        self.db = db
        self.id_ana = None
        self.units = {}

        # method mapping
        self.invoke_for_state.update(OUTPUT=self._on_output)

    def _initialise(self):
        """called during INIT state"""

        # setup analysis
        q = self.db.query("""
        INSERT INTO analysis (expid, tetrode, trialidxstart, trialidxend,
                              algorithm, kind)
        VALUES (%d, %d,
                (SELECT t.trialidx FROM trial t WHERE t.id = %d),
                (SELECT t.trialidx FROM trial t WHERE t.id = %d),
                '%s', '%s');
        SELECT currval('analysis_id_seq');
        """ % (self.mem['id_exp'], self.mem['id_tet'],
               self.mem['trial_ids'][0], self.mem['trial_ids'][0],
               self.mem['algo'], 'SORT'),
                          commit=True)
        self.id_ana = q[0][0]
        self.mem.update(id_ana_out=self.id_ana)

    def _on_output(self):
        """output hook"""

        # exit check
        if self.mem['input_idx'] is None:
            return
            # for all units write spikes and waveform into db
        templates = self.mem['ss'].template_set
        for u in self.mem['sorting']:
            if u not in self.units:
                self.units[u] = self.db.insert_unit(u, self.id_ana, 'single')
                # insert_waveform(db, id_unt, id_trl, wf_data, cutleft=None,
                # snr=None):
            self.db.insert_waveform(self.units[u],
                                    self.mem['input_idx'],
                                    templates[u],
                                    cutleft=self.mem['cutleft'])
            # insert_spiketrain(db, id_trl, id_unt, train):
            if len(self.mem['sorting'][u]) > 0:
                # XXX: START hack for the spike sample unique constraint
                idx = sp.diff(sp.concatenate((
                self.mem['sorting'][u], [self.mem['sorting'][u][-1] + 1]))) \
                > 0
                # XXX: END hack for the spike sample unique constraint
                self.db.insert_spiketrain(self.mem['input_idx'],
                                          self.units[u],
                                          self.mem['sorting'][u][idx])
            # write covmx into db
        # insert_covmx(db, id_ana, mx_data, id_trl=None, kind=None):
        self.db.insert_covmx(self.id_ana,
                             self.mem['ce']._store,
                             self.mem['input_idx'],
                             'noise')
        # update trialidxend
        self.update_trialidxend(self.mem['input_idx'])

    def update_trialidxend(self, id_trl):
        """update the trialidxend after a trial is finished"""

        self.db.query("""
        UPDATE analysis a
        SET trialidxend = (SELECT t.trialidx FROM trial t WHERE t.id = %d)
        WHERE a.id = %d
        """ % (id_trl, self.id_ana),
                      commit=True)


class MunkDebugOutputHandler(NTrodeHandler):
    """database output handler"""

    def __init__(self):
        """debug output and plots"""

        # super
        super(MunkDebugOutputHandler, self).__init__()

        # method mapping
        self.invoke_for_state.update(OUTPUT=self._on_output)

    def _on_output(self):
        """output hook"""

        self.mem['ss'].plot_template_set(show=False)
        self.mem['ss'].plot_sorting(show=True)


class EmailNotificationHandler(NTrodeHandler):
    """notifies by email after execution of plan"""

    def __init__(self, email_targets=[]):
        """
        :Parameters:
            email_server : str
                string of the host address of the email server
            email_targets : list
                list of email addresses
        """

        # super
        super(EmailNotificationHandler, self).__init__()

        # members
        self.email_targets = list(email_targets)

        # method mapping
        self.invoke_for_state.update(OUTPUT=self._on_output)

    def _on_output(self):
        """output hook"""

        if self.mem['input_idx'] is not None:
            return

        print 'sending email notifications..'

        import email, smtplib

        MSG_TEXT = """Hallo Artgenossen,

        neues Sorting am Start: >>> ID:%s <<<

        MfG,
        der SORTER

        :TODO: change this bullshit test message asap
        """ % self.mem['id_ana_out']

        s = smtplib.SMTP_SSL()
        s.connect('smtp.gmail.com', 465)
        s.login('pmeier82@googlemail.com', 'mygoogle')

        for email_target in self.email_targets:
            msg = email.MIMEText(MSG_TEXT)
            msg['Subject'] = '[SORT:BOTMpy] id:%s - %s' % (self.mem['status'])
            msg['From'] = 'pmeier82@googlemail.com'
            msg['To'] = email_target
            s.sendmail(msg['From'], msg['To'], msg.as_string())
            print '..to %s' % email_target

        s.close()
        print 'all emails sent.'

##--- MAIN

if __name__ == '__main__':
    # imports
    import sys, traceback
    from database import MunkSession, CON_GNODE

    # inits
    print sys.argv
    if len(sys.argv) not in (2, 3):
        sys.exit('give 1 argument: 1) id of initialisation and '
                 'optionally True or False for debug state ')
    id_ini = int(sys.argv[1])
    debug = False
    if len(sys.argv) == 3:
        debug = True
    DB = MunkSession(dbconfig=CON_GNODE)
    q = DB.query("""
    SELECT e.name, b.name, t.nr
    FROM analysis a
    JOIN experiment e ON (e.id = a.expid)
    JOIN block b ON (b.id = a.block)
    JOIN tetrode t on (t.id = a.tetrode)
    WHERE a.id = %d
    """ % id_ini)
    if len(q) == 0:
        sys.exit('unknown analysis id: %s' % id_ini)
    exp, blk, tet = q[0]

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
            tet=tet,
            algo='BOTMpy',
            srate=32000.0,
            # internals
            id_ini=id_ini,
            align_at=20,
            dtype=sp.float32,
            debug=debug,
            # handlers
            handlers=[
                (MunkDBInputHandler, {'db':DB}),
                (MunkBOTMHandler, {'db':DB}),
                    {False:(MunkDBOutputHandler, {'db':DB}),
                     True:(MunkDebugOutputHandler, {})}[debug],
                (EmailNotificationHandler,
                     {'email_targets':util.EMAIL_TARGETS})]
        ).run()

    except Exception, ex:
        print '##########################'
        print 'ERROR while processing:', exp, tet
        traceback.print_exception(*sys.exc_info())
        print '##########################'
        print

    finally:
        DB.close()
        del DB

    print
    print 'ALL DONE'
    print
