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
from app.ana_munk.tetrode import Tetrode
from nodes import (PrewhiteningNode, HomoscedasticClusteringNode, PCANode,
                   BOTMNode, SDAbsNode as SDet)
from common import (TimeSeriesCovE, get_cut, snr_maha, mcvec_from_conc,
                    get_aligned_spikes)
from blockstream import (BS3BxpdBlockHeader, BS3BxpdSetupBlock,
                         BS3BxpdDataBlock,
                         BS3SortBlockHeader, BS3SortSetupBlock,
                         BS3SortDataBlock,
                         get_handle, BS3Reader, BXPDProtocolHandler,
                         COVEProtocolHandler, Queue)


##---CONSTANTS

INIT_NOT_STARTED = 0
INIT_DETECTION = 2
INIT_CLUSTER = 4
INIT_SETUP = 6
INIT_SORTING = 10


##---CLASSES

class MunkDummyInputHandler(NTrodeHandler):
    """input handler for blockstream data"""

    def __init__(self):
        """
        :Parameters:
            None
        """

        # super
        super(MunkDummyInputHandler, self).__init__()

        # members
        self.nest = None

        # method mapping
        self.invoke_for_state.update(INPUT=self._on_input)

    def _initialise(self):
        """called during INIT state"""

        self.mem.update(input_idx='MunkBlockstreamInputHandler._initialise')

        self.nest = TimeSeriesCovE(tf_max=self.mem['tf'])
        self.nest.new_chan_set((0, 1, 2, 3))

        # register memory
        self.mem.update(rbuf=None,
                        block_index=None,
                        block_offset=None,
                        nest=self.nest,
                        adaption=False)

    def _on_input(self):
        """called during INPUT state"""

        # reset rbuf
        self.mem.update(rbuf=None)

        # read data
        data = sp.randn(1000, 4)
        data[200:250, :] += sp.vstack(
            [sp.sin(sp.linspace(0, 2 * sp.pi, 50))] * 4).T
        bo, bi = 1000, 1000

        self.mem.update(rbuf=data,
                        input_idx=bi,
                        block_offset=bo)


class MunkBlockstreamInputHandler(NTrodeHandler):
    """input handler for blockstream data"""

    def __init__(self):
        """
        :Parameters:
            None
        """

        # super
        super(MunkBlockstreamInputHandler, self).__init__()

        # members
        self.iq_bxpd = None
        self.iq_cove = None
        self.bxpd_reader = None
        self.cove_reader = None
        self.setup_block = None
        self._chans = []
        self.nest = None
        self.mute = False

        self.cnt = 0

        # method mapping
        self.invoke_for_state.update(INPUT=self._on_input)

    def _initialise(self):
        """called during INIT state"""

        if self.mem['debug']:
            print 'MunkBlockstreamInputHandler._initialise enter'

        self.mem.update(input_idx='MunkBlockstreamInputHandler._initialise')

        self.iq_bxpd = Queue()
        self.iq_cove = Queue()

        self.bxpd_reader = BS3Reader(BXPDProtocolHandler, self.iq_bxpd,
                                     verbose=False, #self.mem['debug'],
                                     ident='SORT[%d] BXPD' % self.mem['tet'])
        self.bxpd_reader.start()
        self.cove_reader = BS3Reader(COVEProtocolHandler, self.iq_cove,
                                     verbose=False, #self.mem['debug'],
                                     ident='SORT[%d] COVE' % self.mem['tet'])
        self.cove_reader.start()
        print 'BlockStream Reader Threads started!'

        self.nest = TimeSeriesCovE(tf_max=self.mem['tf'])
        self.nest.new_chan_set(tuple(xrange(self.mem['nc'])))

        # register memory
        self.mem.update(rbuf=None,
                        block_index=None,
                        block_offset=None,
                        nest=self.nest,
                        adaption=False)

        if self.mem['debug']:
            print 'MunkBlockstreamInputHandler._initialise exit'

    def _on_input(self):
        """called during INPUT state"""

        #        self.cnt += 1
        #        if self.cnt > 20000:
        #            print 'CNT =', self.cnt, self.mem['input_idx']
        #            if self.mem['ini_state'] == INIT_SORTING:
        #                self.mem['input_idx'] = None
        #                return

        # exit/mute check
        if self.mem['ini_state'] == INIT_CLUSTER:
            self.bxpd_reader.mute(True)
            self.cove_reader.mute(True)
        if self.mem['ini_state'] == INIT_SORTING:
            self.bxpd_reader.mute(False)
            self.cove_reader.mute(False)

        # reset rbuf
        self.mem.update(rbuf=None)

        #        # queue flush after initialisation
        #        if self.mem['ini_state'] == INIT_SETUP:
        #            while not self.iq_bxpd.empty():
        #                item = self.iq_bxpd.get()
        ##                del item
        #            while not self.iq_cove.empty():
        #                item = self.iq_cove.get()
        ##                del item
        #            self.mem['ini_state'] = INIT_SORTING
        #            return

        # read data
        data_lst = []
        bo, bi = None, None
        if self.iq_bxpd.qsize() > 10:
            print 'qsize:', self.iq_bxpd.qsize()
        while True:
            block_header, protocol_block = self.iq_bxpd.get()
            if isinstance(protocol_block, BS3BxpdSetupBlock):
                self.setup_block = protocol_block
                chan_nrs = self.setup_block.group_lst[self.mem['tet']][-1]
                self._chans = []
                for chan_nr in chan_nrs:
                    self._chans.append(
                        self.setup_block.anchan_index_mapping[chan_nr])
                print 'now listening to:',
                self.setup_block.group_lst[self.mem['tet']][1]
            if isinstance(protocol_block, BS3BxpdDataBlock):
                if bo is None:
                    sr_idx = self.setup_block.anchan_lst[self._chans[0]][1]
                    bo = protocol_block.srate_lst[sr_idx]
                if bi is None:
                    bi = block_header.block_index
                data = []
                for c in self._chans:
                    data.append(protocol_block.anchan_lst[c])
                data = sp.vstack(data).T
                data_lst.append(data)
                #            self.iq_bxpd.task_done()
            #            del block_header, protocol_block
            if self.iq_bxpd.empty():
                break
        if len(data_lst) > 0:
            data_lst = sp.vstack(data_lst).astype(self.mem['dtype'])
        else:
            data_lst = None

        while not self.iq_cove.empty():
            block_header, protocol_block = self.iq_cove.get()
            if protocol_block.data_lst[0] != self.mem['tet']:
            #                del block_header, protocol_block
                continue
            new_xcorrs = protocol_block.data_lst[-2]
            for i in xrange(self.mem['nc']):
                for j in xrange(i, self.mem['nc']):
                    self.nest._store[i, j] = new_xcorrs[i * self.mem['nc'] +
                                                        j]
            if not self.nest._is_initialised:
                self.nest._is_initialised = True
                #            self.mem.update(nest=self.nest)
            self.mem['adaption'] = True
            print 'adaption incoming'
            #            self.iq_cove.task_done()
        #            del block_header, protocol_block

        self.mem.update(rbuf=data_lst,
                        block_index=bi,
                        block_offset=bo)

    def _on_finalize(self):
        """finalize event"""

        self.cove_reader.stop()
        self.cove_reader.join()
        self.bxpd_reader.stop()
        self.bxpd_reader.join()
        print 'BlockStream Reader Threads stopped ! '


class MunkOnlineInitialisationHandler(NTrodeHandler):
    """initialisation handler"""

    def __init__(self):#, detector_cls, detector_cls_kwargs):
        """
        :Parameters:
            detector_cls : ThresholdDetectorNode
                class to spawn the spikedetector from
            detector_cls_kwargs : dict
                dictionary with spawning keyword arguments for the spike
                detector.
        """

        # super
        super(MunkOnlineInitialisationHandler, self).__init__()

        # members
        self.sdet = None
        self.spks = []
        self.spks_reduced = None
        self.units = None

        # method mapping
        self.invoke_for_state.update(PROCESS=self._on_process)

    def _initialise(self):
        """called during INIT"""

        if self.mem['debug']:
            print 'MunkOnlineInitialisationHandler._initialise enter'

        # setup members
        self.sdet = SDet(threshold_factor=self.mem['th_det'],
                         tf=self.mem['tf'])
        #                         min_dist=int(self.mem['tf'] * 0.5))

        if self.mem['debug']:
            print 'MunkOnlineInitialisationHandler._initialise exit'

    def _on_process(self):
        """accumulates and then clusters detected spike to produce the initial
        template set"""

        # exit checks
        if self.mem['rbuf'] == None:
            return
        if self.mem['ini_state'] != INIT_DETECTION or self.mem[
                                                      'ini_state'] !=\
                                                      INIT_CLUSTER:
            if self.mem['ini_state'] != INIT_CLUSTER:
                self.mem['ini_state'] = INIT_DETECTION

        # detection step
        if self.mem['ini_state'] == INIT_DETECTION:
            self._detection_step()
        elif self.mem['ini_state'] == INIT_CLUSTER:
            self._cluster_step()
            self.invoke_for_state.pop('PROCESS')
            print 'poping initialisation on_process'
            print self.invoke_for_state

    def _detection_step(self):
        """spike detection over current buffer"""

        # spike detection
        self.sdet(self.mem['rbuf'])
        cut = get_cut(self.mem['tf'])

        # alignment and save
        spks, st = get_aligned_spikes(self.mem['rbuf'],
                                      self.sdet.events,
                                      cut,
                                      align_at=self.mem['align_at'],
                                      mc=False,
                                      kind='min')
        if spks.size > 0:
            self.spks.append(spks)
        self.mem.update(sorting={65535:st})

        # debug out
        new_spks = len(spks)
        old_spks = sum([item.shape[0] for item in self.spks])
        if self.mem['debug']:
            if new_spks > 0:
                print 'found %d spikes [%d]' % (new_spks, old_spks)

        # cluster step?
        if old_spks + new_spks > self.mem['ini_nspks']:
            self.mem['ini_state'] = INIT_CLUSTER

    def _cluster_step(self):
        """start inititialisation clustering"""

        if self.mem['debug']:
            from plot import P, waveforms

        # save all spikes in one matrix
        self.spks = sp.vstack(self.spks)

        # assert we have a cmx, else use identity
        if not self.mem['nest'].is_initialised():
            print 'NEST not initialized! resorting to identity!'
            self.mem['nest'].update(sp.randn(10000, self.mem['nc']))
        cmx = self.mem['nest'].get_cmx(tf=self.mem['tf'],
                                       chan_set=(0, 1, 2, 3))
        icmx = self.mem['nest'].get_icmx(tf=self.mem['tf'],
                                         chan_set=(0, 1, 2, 3))
        if self.mem['debug']:
            P.matshow(cmx)

        # check snr of spikes found
        self.snr = snr_maha(self.spks, icmx)
        good_spks = self.snr > self.mem['th_snr']
        if self.mem['debug']:
            print 'found', self.spks.shape[0], 'spikes'
            print 'keeping', good_spks.sum()
            P.figure()
            P.hist(self.snr, bins=100)
            P.axvline(self.mem['th_snr'], c='y')
        self.spks = self.spks[good_spks]
        self.snr = self.snr[good_spks]
        if self.mem['debug']:
            print 'keeping', self.spks.shape[0], 'spikes with SNR > ',
            self.mem['th_snr']
            # build processing chain
        if self.mem['debug']:
            print 'starting to prewhiten w.r.t. noise..'
        prw = PrewhiteningNode(ncov=cmx)
        pca = PCANode(output_dim=self.mem['th_pca'])
        clu = HomoscedasticClusteringNode(clus_type='gmm',
                                          crange=range(1, 10),
                                          sigma_factor=4.0,
                                          maxiter=128,
                                          repeats=5,
                                          dtype=self.mem['dtype'],
                                          debug=self.mem['debug'],
                                          weights_uniform=False)

        # pca into desired resolution
        self.spks_reduced = pca(prw(self.spks))
        clu(self.spks_reduced)
        if self.mem['debug']:
            print 'explaining %.5f of total (prewhitened) variance with %d'\
                  'components' % (pca.explained_variance * 100,
                                  pca.output_dim)
            print 'starting to cluster..'
            clu.plot(self.spks_reduced, views=6, show=False)

        # build templates
        nunits = int(clu.labels.max() + 1)
        if self.mem['debug']:
            print 'creating units'
        self.units = sp.zeros((nunits, self.spks.shape[1]))
        for u in xrange(nunits):
            self.units[u] = self.spks[clu.labels == u].mean(axis=0)

        # debug plotting
        if self.mem['debug'] is True:
            clu.plot(self.spks_reduced, views=3, show=False)
            wf_data = {}
            for u in xrange(self.units.shape[0]):
                wf_data[u] = self.spks[clu.labels == u]
            waveforms(wf_data,
                      tf=self.mem['tf'],
                      plot_mean=True,
                      plot_separate=True,
                      show=True)
            P.show()

        # save covariance and templates
        templates = sp.zeros(
            (self.units.shape[0], self.mem['tf'], self.mem['nc']))
        for u in xrange(self.units.shape[0]):
            templates[u] = mcvec_from_conc(self.units[u], nc=self.mem['nc'])
        self.mem.update(templates=templates)
        self.mem['ini_state'] = INIT_SETUP


class MunkBOTMHandler(NTrodeHandler):
    """handler applying the BOTM spikesorting to the data"""

    def __init__(self):
        """
        :Parameters:
            None
        """

        # super
        super(MunkBOTMHandler, self).__init__()

        # members
        self.ss = None

        # method mapping
        self.invoke_for_state.update(PROCESS=self._on_process)

    def _initialise(self):
        """setup the sorter"""

        if self.mem['debug']:
            print 'MunkBOTMHandler._initialise enter'

        self.mem.update(input_idx='MunkBOTMHandler._initialise')
        if self.mem['ini_state'] < INIT_SETUP:
            return

        if self.mem['debug']:
            print 'building sorter'

        ## sorting node
        #templates,
        #chan_set=(0, 1, 2, 3),
        #ce=None,
        #rb_cap=350,
        #adapt_templates=0,
        #learn_noise=True,
        #chunk_size=100000,
        #use_history=False,
        #use_clib=False,
        #debug=False,
        #dtype=None,
        ## bss node
        #ovlp_taus=[-2, 0, 2],
        #spk_pr=1e-6,
        #noi_pr=1e1,
        self.ss = BOTMNode(self.mem['templates'],
                           chan_set=(0, 1, 2, 3),
                           ce=self.mem['nest'],
                           rb_cap=self.mem['ss_buf'],
                           adapt_templates=self.mem['align_at'],
                           learn_noise=False,
                           chunk_size=int(self.mem['srate']),
                           use_history=True,
                           use_clib=True,
                           debug=self.mem['debug'],
                           ovlp_taus=[-2, 0, 2], )
        self.mem.update(ss=self.ss)

        self.mem['ini_state'] = INIT_SORTING

        if self.mem['debug']:
            print 'MunkBOTMHandler._initialise exit'

    def _on_process(self):
        """perform sorting"""

        # exit check
        if self.mem['input_idx'] is None:
            return
        if self.mem['rbuf'] == None:
            return
        if self.mem['ini_state'] < INIT_SORTING:
            if self.mem['ini_state'] == INIT_SETUP:
                self._initialise()
            else:
                return

        # chunked sorting
        self.ss(self.mem['rbuf'])
        self.mem.update(sorting=self.ss.rval)
        if self.mem['debug']:
            print sum([len(self.ss.rval[k]) for k in self.ss.rval])

        # check for adaption
        if self.mem['adaption'] is True:
            if 'adaption_done' in self.mem:
                ret = self.ss._check_internals_par_collect()
                if ret is True:
                    self.mem.pop('adaption_done')
            else:
                self.ss._check_internals_par()
                self.mem['adaption_done'] = False


class MunkBlockstreamOutputHandler(NTrodeHandler):
    """output handler using the blockstream protocol"""

    def __init__(self):
        """
        :Parameters:
            None
        """

        # super
        super(MunkBlockstreamOutputHandler, self).__init__()

        # members
        self._blkstr = None
        self.writer_id = None
        self.preamble = None

        # method mapping
        self.invoke_for_state.update(OUTPUT=self._on_output)

    def _initialise(self):
        """called during INIT state"""

        if self.mem['debug']:
            print 'MunkBlockstreamOutputHandler._initialise enter'

        # initialise the blockstream library
        self._blkstr = get_handle()
        self.writer_id = self._blkstr.startWriter(
            'SORT_SORT[%s]' % self.mem['tet'])
        group_lst = [self.mem['tet'],
                     self.mem['nc'],
                     self.mem['tf'],
                     get_cut(self.mem['tf'])[0],
                     sp.zeros((self.mem['tf'] ** 2 * self.mem['nc'] ** 2)),
            []]
        det_preamble = BS3SortSetupBlock(BS3SortBlockHeader(0), [group_lst])
        det_preamble.send_me(self._blkstr, self.writer_id)

        if self.mem['debug']:
            print 'MunkBlockstreamOutputHandler._initialise exit'

    def _on_output(self):
        """output hook"""

        # exit check
        if self.mem['input_idx'] is None:
            return
        if self.mem['rbuf'] == None:
            return
        if self.mem['ini_state'] < INIT_DETECTION:
            return

        # preamble check
        if self.mem['ini_state'] >= INIT_SORTING:
            if self.preamble is None or self.mem['adaption'] is True:
                self.build_preamble()
                if self.mem['adaption'] is True:
                    self.mem['adaption'] = False

        # build new sorting block
        event_lst = []
        for u in self.mem['sorting']:
            for t in self.mem['sorting'][u]:
            #                if t < 0:
            #                    print 'bad timesample for event in block: t
            # = % s' % t
            #                    t = 0
            #                print t + self.mem['block_offset']
                event_lst.append((
                    self.mem['tet'], u, t + self.mem['block_offset'], 0, 0,
                    0))
        if len(event_lst) > 0:
            block = BS3SortDataBlock(BS3SortBlockHeader(1), event_lst)
            block.send_me(self._blkstr, self.writer_id)

    def build_preamble(self):
        nunits = self.mem['ss'].nfilter
        temp_set = self.mem['ss'].template_set
        filt_set = self.mem['ss'].filter_set
        units = [(filt_set[u], temp_set[u], 1, 0, 0)
        for u in xrange(nunits)]
        group_lst = [self.mem['tet'],
                     self.mem['nc'],
                     self.mem['tf'],
                     get_cut(self.mem['tf'])[0],
                     self.mem['nest'].get_cmx(tf=self.mem['tf'],
                                              chan_set=(0, 1, 2, 3)),
                     units]
        self.preamble = BS3SortSetupBlock(BS3SortBlockHeader(0), [group_lst])
        self.preamble.send_me(self._blkstr, self.writer_id)

##--- MAIN

if __name__ == '__main__':
    # imports
    import sys, traceback

    # inits
    print 'callstr', sys.argv
    exp = 'L020'
    blk = 'a'
    tet = 0

    if len(sys.argv) != 5:
        sys.exit('give 4 arguments ! exp, blk, tet, debug')
    exp = str(sys.argv[1])
    blk = str(sys.argv[2]).lower()
    tet = int(sys.argv[3])
    debug = str(sys.argv[4])
    if debug.lower() == 'true':
        debug = True
    else:
        debug = False

    # start the tetrode
    try:
        print
        print '########################'
        print 'starting Tetrode'
        print 'exp:', exp, 'blk:', blk, 'tet:', tet
        print 'mode: online::blockstream!'
        print '########################'
        print

        T = Tetrode(
            name='BlockstreamTetrode: % s' % tet,
            exp=exp,
            blk=blk,
            tet=tet,
            algo='onlineBOTMpy',
            srate=32000.0,
            tf=30,
            nc=4,
            # internals
            align_at=10,
            ini_state=INIT_NOT_STARTED,
            ini_nspks=1000,
            ss_buf=350,
            th_pca=6,
            th_snr=1.0,
            th_det=3.5,
            dtype=sp.float32,
            debug=debug,
            # handlers
            handlers=[
                (MunkBlockstreamInputHandler, {}),
                #                (MunkDummyInputHandler, {}),
                (MunkOnlineInitialisationHandler, {}),
                (MunkBOTMHandler, {}),
                (MunkBlockstreamOutputHandler, {}),
                ]
        )
        T.run()

    #        print 'testing profile'
    #        import cProfile
    #        cProfile.runctx("""Tetrode(name='BlockstreamTetrode: % s' %
    # tet, exp=exp, blk=blk, tet=tet, algo='onlineBOTMpy', srate=32000.0,
    # tf=30, nc=4, align_at=10, ini_state=INIT_NOT_STARTED, ini_nspks=1000,
    # ss_buf=350, th_pca=6, th_snr=0.7, th_det=3.5, dtype=sp.float32,
    # debug=debug, handlers=[(MunkBlockstreamInputHandler, {}),
    # (MunkOnlineInitialisationHandler, {}), (MunkBOTMHandler, {}),
    # (MunkBlockstreamOutputHandler, {}), ]).run()""", globals=globals(),
    # locals=locals())

    except Exception, ex:
        print '##########################'
        print 'ERROR while processing:', exp, blk, tet
        traceback.print_exception(*sys.exc_info())
        print '##########################'
        print

        print 'finalizing Tetrode'
        #        T.finalise()
        print

    print
    print 'ALL DONE'
    print
