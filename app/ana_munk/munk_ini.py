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


"""NTrode handlers for the Munk data analysis for establishing an initial sorting"""
__docformat__ = 'restructuredtext'


##--- IMPORTS

import scipy as sp
from ntrode import NTrodeHandler, NTrodeError
from nodes import PrewhiteningNode, HomoscedasticClusteringNode, PCANode
from database import MunkSession, CON_GNODE
from tetrode import Tetrode
from common import (TimeSeriesCovE, get_cut, epochs_from_spiketrain,
                    invert_epochs, merge_epochs, snr_maha, mcvec_from_conc,
                    get_aligned_spikes)
import util


##--- CLASSES

class MunkIniDBHandler(NTrodeHandler):
    """initialisation handler"""

    def __init__(self, db=None, mean_correct=False):
        """
        :Parameters:
            db : MunkPostgresDB
                database handle
            mean_correct : bool
                If True, mean correct the data before processing it.
        """

        # super
        super(MunkIniDBHandler, self).__init__()

        # members
        self.db = db
        self.mean_correct = bool(mean_correct)
        self.data = {}
        self.trial_ids = None
        self.id_exp = None
        self.id_blk = None
        self.id_tet = None
        self.id_ana_mu = None
        self.id_unt_mu = None
        self.id_ana_ini = None
        self.ndet = None
        self.prw = None
        self.pca = None
        self.cls = None
        self.spks = None
        self.spks_prw = None
        self.spks_info = None
        self.units = None

        # method mapping
        self.invoke_for_state.update(INPUT=self._on_input,
                                     PROCESS=self._on_process)

    def _initialise(self):
        """called during INIT state
        
        will basically check all input parameters and establish database
        connectivity.
        """

        # check if there is an analysis with the passed id
        try:
            self.id_ana_mu = int(self.mem['ana'])
        except TypeError, KeyError:
            raise NTrodeError('no analysis id found. '
                              'use: ana=<id> in Tetrode constructor!')

        # get relevant info for analysis item
        q = self.db.query("""
        SELECT 
          a.expid, 
          a.block, 
          a.tetrode,
          a.trialidxstart,
          a.trialidxend
        FROM 
          analysis a
        WHERE 
          a.id = %d
        """ % self.id_ana_mu)
        if q is None or q == []:
            raise NTrodeError('did not found an analysis for id: %d' % self.id_ana_mu)
        self.id_exp = q[0][0]
        self.id_blk = q[0][1]
        self.id_tet = q[0][2]
        trls = q[0][3]
        trle = q[0][4]

        # elaborate the trial range to initialise over
        if 'ini_range' not in self.mem:
            raise NTrodeError('no ini_range given. '
                              'use: ini_range=<int> in Tetrode constructor!')
        self.mem.update(inirange=min(trle - trls, int(self.mem['ini_range'])))

        # build trial id list
        self.trial_ids = self.db.get_trial_range(self.id_blk,
                                                 include_error_trials=False,
                                                 limit=self.mem['ini_range'])
        if len(self.trial_ids) == 0:
            raise NTrodeError('empty trial list')

        # elaborate the unit id for the multi unit spiketrain
        self.id_unt_mu = self.db.get_units_for_analysis(self.id_ana_mu)
        if len(self.id_unt_mu) == 0:
            raise NTrodeError('did not found a matching multiunit spiketrain '
                              'for analysis: %d' % self.id_ana_mu)
        self.id_unt_mu = self.id_unt_mu[0]

        # setup members
        self.ndet = TimeSeriesCovE(tf_max=self.mem['tf'],
                                   nc=4,
                                   dtype=self.mem['dtype'])
        self.ndet.new_chan_set((0, 1, 2, 3))

        # algorithm string
        self.mem['algo'] = self.mem['algo'] % (self.mem['ana'],
                                               self.mem['th_pca'],
                                               self.mem['th_snr'])
        print 'INIT: %s' % self.mem['algo']

        # establish data management
        # we want to store all the spikes, and for each spike the information
        # we will need to put it in the right spike train in the right trial
        #
        # so we will have:
        # spks       : one large matrix holding the spikes in the rows
        # spks_info  : matrix with trial_id, sample, label in the rows

        if self.mem['debug'] is True:
            self.invoke_for_state.update(OUTPUT=self._on_output_debug)
            print '*** not writing to DB, showing pictures instead -.- ***'
        else:
            self.invoke_for_state.update(OUTPUT=self._on_output)

    def _on_input(self):
        """called during INPUT state
        
        will iterate over the trials and build the spike set to cluster on
        """

        if self.mem['debug'] is True:
            from plot import P, waveforms

        print 'loading data trials..'
        for id_trl in self.trial_ids:
            trial_data = None
            try:
                trial_data = self.db.get_tetrode_data(id_trl,
                                                      self.id_tet,
                                                      sp.dtype(self.mem['dtype']))
                if self.mean_correct is True:
                    trial_data -= trial_data.mean(axis=0)
                print '\tprocessed %s' % self.db.get_fname_for_id(id_trl)
                self.data[id_trl] = trial_data
            except Exception, e:
                raise NTrodeError('error processing %s\n%s' %
                                  (self.db.get_fname_for_id(id_trl), e))
            finally:
                del trial_data
        print 'done.'

        print 'retrieving raw multiunit spike set @tf=%d' % self.mem['tf']
        self.spks_info = []
        self.spks = []
        for id_trl in self.trial_ids:
            trial_st = None
            try:
                trial_st = self.db.get_unit_data(self.id_unt_mu, id_trl)['spiketrain']
                if trial_st.size == 0:
                    print '\t no spiketrain for %s' % self.db.get_fname_for_id(id_trl)
                    continue
                cut = get_cut(self.mem['tf'])
                trial_spks, trial_st = get_aligned_spikes(
                    self.data[id_trl],
                    trial_st,
                    self.mem['tf'],
                    align_at=self.mem['align_at'],
                    mc=False,
                    kind='min')
                nep = epochs_from_spiketrain(trial_st, cut, end=self.data[id_trl].shape[0])
                nep = invert_epochs(nep, end=self.data[id_trl].shape[0])
                nep = merge_epochs(nep)
                self.ndet.update(self.data[id_trl], epochs=nep)
                self.spks.append(trial_spks)
                self.spks_info.append(sp.vstack([[id_trl] * trial_st.size, trial_st]).T)
                print '\tprocessed %s' % self.db.get_fname_for_id(id_trl)
            except Exception, e:
                raise NTrodeError('error processing %s\n%s' %
                                  (self.db.get_fname_for_id(id_trl), e))
            finally:
                del trial_st
        self.spks_info = sp.vstack(self.spks_info)
        self.spks = sp.vstack(self.spks)
        print 'found %d spikes in total' % self.spks.shape[0]
        if self.mem['debug'] is True:
            waveforms({0:self.spks}, show=False, title='ALIGNED SPIKES')
        print 'done.'

        print 'checking SNR of spikes'
        snr = snr_maha(self.spks, self.ndet.get_icmx(tf=self.mem['tf'],
                                                     chan_set=(0, 1, 2, 3)))
        if self.mem['debug'] is True:
            P.figure()
            P.hist(snr, bins=100)
            P.axvline(self.mem['th_snr'], c='y')
        good_spks = snr > self.mem['th_snr']
        n_spks = self.spks.shape[0]
        self.spks = self.spks[good_spks]
        self.spks_info = self.spks_info[good_spks].astype(int)
        print 'keeping %d of %d spikes with SNR > %f' % (self.spks.shape[0],
                                                         n_spks, self.mem['th_snr'])

        print 'starting to prewhiten w.r.t. noise..'
        prw = PrewhiteningNode(ncov=self.ndet.get_cmx(tf=self.mem['tf'], chan_set=(0, 1, 2, 3)))
        self.spks_prw = prw(self.spks)
        print 'done.'

    def _on_process(self):
        """perform sorting"""

        if self.mem['debug'] is True:
            from plot import waveforms, cluster_projection

        # build processing chain
        pca = PCANode(output_dim=self.mem['th_pca'])
        clu = HomoscedasticClusteringNode(clus_type='gmm',
                                          sigma_factor=5,
                                          maxiter=128,
                                          repeats=4,
                                          dtype=self.mem['dtype'],
                                          debug=self.mem['debug'],
                                          weights_uniform=False)
        # pca into desired resolution
        print 'reducing dimensionality..'
        sp_pca = pca(self.spks_prw)
        print 'explaining %.5f of total variance with %d components' % \
            (pca.explained_variance * 100, pca.output_dim)

        print 'starting to cluster..'
        clu(sp_pca)
        self.spks_info = sp.vstack([
            self.spks_info[:, 0],
            self.spks_info[:, 1],
            clu.labels
        ]).T
        nunits = int(clu.labels.max() + 1)
        print 'done.'

        print 'creating units'
        self.units = sp.zeros((nunits, self.spks.shape[1]))
        for i in xrange(nunits):
            self.units[i] = self.spks[clu.labels == i].mean(axis=0)
        print 'done.'

        if self.mem['debug'] is True:
            from scipy import linalg as sp_la
            clu.plot(sp_pca, views=3, show=False)
            wf_data = {}
            sp_data = {}
            mcd = sp.zeros((self.units.shape[0] - 1, self.units.shape[0] - 1))
            for u in xrange(self.units.shape[0]):
                wf_data[u] = self.spks[clu.labels == u]
                sp_data[u] = self.spks_prw[clu.labels == u]
            for u in xrange(self.units.shape[0] - 1):
                for u1 in xrange(u + 1, self.units.shape[0]):
                    mean_u = sp_data[u].mean(0)
                    mean_u1 = sp_data[u1].mean(0)
                    con = mean_u - mean_u1
                    con /= sp_la.norm(con)
                    mcd[u, u1 - 1] = sp.absolute(sp.dot(mean_u, con) -
                                                 sp.dot(mean_u1, con))
            waveforms(wf_data,
                      tf=self.mem['tf'],
                      plot_mean=True,
                      plot_separate=True,
                      show=False)
            try:
                cluster_projection(sp_data, show=False)
            except:
                print 'only one unit, omitting cluster projection for only one unit!!'
            print
            print mcd
            print

    def _on_output_debug(self):
        """output hook"""

        print 'DEBUG OUTPUT HOOK, no DB spamming'
        from plot import P
        P.show()

        # ask for saving to db
        answer = raw_input('to save initialisation to database please write "yes": ')
        if answer.lower() == 'yes':
            self._on_output()

        # we are finished !!
        self.mem.update(input_idx=None)

    def _on_output(self):
        """output hook"""

        print 'setting up analysis item in database'
        trial_start = self.db.get_trial_range(self.id_blk, limit=1)[0]
        trial_end = self.db.get_trial_range(self.id_blk, limit=self.mem['ini_range'])[-1]
        q = self.db.query("""
        INSERT INTO analysis (expid, block, tetrode, trialidxstart, trialidxend,
                              algorithm, kind)
        VALUES (%d, %d, %d,
                (SELECT t.trialidx FROM trial t WHERE t.id = %d),
                (SELECT t.trialidx FROM trial t WHERE t.id = %d),
                '%s', '%s');
        SELECT currval('analysis_id_seq');
        """ % (self.id_exp, self.id_blk, self.id_tet, trial_start, trial_end,
               self.mem['algo'], 'INIT'),
        commit=True)
        self.id_ana_ini = q[0][0]
        print 'done.'

        print 'saving single unit clustering'
        nunits = self.units.shape[0]
        cl = get_cut(self.mem['tf'])[0]
        for u in xrange(nunits):
            print '\tunit %d' % u
            id_unt_u = self.db.insert_unit(u, self.id_ana_ini, 'single-init')
            for id_trl in self.trial_ids:
                # save template for this trial and unit
                # insert_waveform(db, id_unt, id_trl, wf_data, cutleft=None, snr=None):
                snr = snr_maha(sp.atleast_2d(self.units[u]),
                               self.ndet.get_icmx(tf=self.mem['tf'],
                                                  chan_set=(0, 1, 2, 3)))[0]
                self.db.insert_waveform(id_unt_u, id_trl,
                                        mcvec_from_conc(self.units[u],
                                                        nc=self.mem['nc']),
                                        cutleft=cl, snr=snr)
                # save spiketrain for this trial and unit
                st_idx = self.spks_info[:, 0] == id_trl
                st_idx *= self.spks_info[:, 2] == u
                st = self.spks_info[st_idx, 1]
                self.db.insert_spiketrain(id_trl, id_unt_u, st)
        print 'done.'

        print 'saving covariance matrix'
        # insert_covmx(db, id_ana, mx_data, id_trl=None, kind=None):
        self.db.insert_covmx(self.id_ana_ini, self.ndet._store,
                             self.trial_ids[0],
                             'noise')
        print 'done.'

        # we are finished!
        self.mem.update(input_idx=None)



##--- MAIN

if __name__ == '__main__':

    # imports
    import sys, traceback

    # db session
    DB = MunkSession(dbconfig=CON_GNODE)
    DB.connect()

    # inits
    debug = False
    if len(sys.argv) == 2 or len(sys.argv) == 3:
        ana = int(sys.argv[1])
        q = DB.query("""
        SELECT e.name, b.name, t.nr
        FROM analysis a
        JOIN experiment e ON (e.id=a.expid)
        JOIN block b ON (b.id=a.block)
        JOIN tetrode t ON (t.id=a.tetrode)
        WHERE a.id = %d
        """ % ana)
        if q is None or q == []:
            sys.exit('did not found an analysis for id: %d' % ana)
        exp = str(q[0][0])
        blk = str(q[0][1]).lower()
        tet = int(q[0][2])
        if len(sys.argv) == 3:
            debug = True
    elif len(sys.argv) == 4 or len(sys.argv) == 5:
        exp = str(sys.argv[1])
        blk = str(sys.argv[2]).lower()
        tet = int(sys.argv[3])
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
        if id_tet is None:
            sys.exit('no matching tetrode for: %s-%s-%s' % (exp, blk, tet))
        if len(sys.argv) == 5:
            debug = True
    else:
        sys.exit('give exactly 1 or 3 arguments:1) analysis id or 2) '
                 'exp, block, tetrode !!!')

    # start the tetrode
    try:

        print
        print
        print '########################'
        print 'starting initialisation clustering'
        print 'ana:', ana
        print 'exp:', exp, 'blk:', blk, 'tet:', tet
        print '########################'
        print
        print

        Tetrode(
            debug=debug,
            name='INI:%s %s %s %s' % (ana, exp, blk, tet),
            exp=exp,
            blk=blk,
            tet=tet,
            algo='MU(%s)::GMM(1-15)<PCA(%s)<PRW<SNR(%s)',
            # internals
            srate=32000.0,
            tf=65,
            nc=4,
            dtype=sp.float32,
            # handlers
            handlers=[
                (MunkIniDBHandler, {'db':DB}),
            ],
            # additionals
            ana=ana,
            ini_range=7,
            th_pca=6,
            th_snr=1.0,
            align_at=20,
        ).run()

    except Exception, ex:

        print
        print
        print '##########################'
        print 'ERROR while processing:', exp, blk, tet
        traceback.print_exception(*sys.exc_info())
        print '##########################'
        print
        print

    finally:

        DB.close()
        del DB

    print
    print 'ALL DONE'
    print
