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
from blockstream import (BS3BxpdBlockHeader, BS3BxpdSetupBlock, BS3BxpdDataBlock,
                         BS3SortBlockHeader, BS3SortSetupBlock, BS3SortDataBlock,
                         get_handle, BS3Reader, BXPDProtocolHandler,
                         COVEProtocolHandler, Queue)
from tables import openFile


##---CONSTANTS

INIT_NOT_STARTED = 0
INIT_DETECTION = 2
INIT_CLUSTER = 4
INIT_SETUP = 6
INIT_SORTING = 10


##---CLASSES

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
        self.setup_block = None

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

        if self.mem['ini_state'] < INIT_SORTING and self.mute is False:
            self.bxpd_reader.mute(True)
            self.cove_reader.mute(True)
            self.mute = True
        if self.mem['ini_state'] == INIT_SORTING and self.mute is True:
            self.bxpd_reader.mute(False)
            self.cove_reader.mute(False)
            self.mute = False

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
                    self._chans.append(self.setup_block.anchan_index_mapping[chan_nr])
                print 'now listening to:', self.setup_block.group_lst[self.mem['tet']][1]
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
            self.nest._reset()
            new_xcorrs = protocol_block.data_lst[-2]
            for i in xrange(self.mem['nc']):
                for j in xrange(i, self.mem['nc']):
                    self.nest._store[i, j] = new_xcorrs[i * self.mem['nc'] + j]
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


class MunkHDF5InitialisationHandler(NTrodeHandler):
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
        super(MunkHDF5InitialisationHandler, self).__init__()

    def _initialise(self):
        """called during INIT"""

        if self.mem['debug']:
            print 'MunkHDF5InitialisationHandler._initialise enter'

        # setup members
        # arc = openFile('C:\\Users\\phil\\Development\\SpiDAQ\\SpikePy\\app\\ana_munk\\eval\\bmark_eval_n7_snr2_10min.h5', 'r')
        print 'opening archive'
        arc = openFile('E:\\Media\\NeuroMeter\\bmark_eval_n7_snr2_10min.h5', 'r')
        wfs = {}
        sts = {}
        uid = 0
        end = 32000 * 20
        print 'reading data'
        signal = arc.getNode('/data').read()[:]
        print signal
        abs_max_val = max(signal.max(), -signal.min())
        print '2:', abs_max_val
        import_factor = 2048.0 / abs_max_val
        print '3:', import_factor
        signal *= import_factor
        print '4'
        wf_start = 155
        print 'reading units'
        for unit in arc.getNode('/groundtruth'):
            print wf_start, wf_start + self.mem['tf']
            wfs[uid] = sp.asarray(unit.waveform.read()[wf_start:wf_start + self.mem['tf'], :])
            wfs[uid] *= import_factor
            sts[uid] = sp.asarray(unit.train.read())
            sts[uid] = sts[uid][sts[uid] < end]
            uid += 1
        print 'closing archive'
        arc.close()
        del arc
        from common import epochs_from_spiketrain_set
        ep = epochs_from_spiketrain_set(sts, cut=(0, self.mem['tf']), end=end)
        nep = ep['noise']
        self.mem['nest'].update(signal[:end, :], epochs=nep)
        del ep, nep, signal

        # save covariance and templates
        templates = sp.zeros((len(wfs), self.mem['tf'], self.mem['nc']))
        for u in xrange(len(wfs)):
            templates[u] = wfs[u]
        self.mem.update(templates=templates)
        del wfs
        self.mem['ini_state'] = INIT_SETUP
        print 'done initialising'

        if self.mem['debug']:
            print 'MunkHDF5InitialisationHandler._initialise exit'


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
        self.ss = BOTMNode(templates=self.mem['templates'],
                           chan_set=(0, 1, 2, 3),
                           ce=self.mem['nest'],
                           rb_cap=self.mem['ss_buf'],
                           adapt_templates=self.mem['align_at'],
                           learn_noise=False,
                           chunk_size=int(self.mem['srate']),
                           use_history=True,
                           use_clib=True,
                           debug=self.mem['debug'],
                           ovlp_taus=[-2, 0, 2],)
        self.mem.update(ss=self.ss)

        self.mem['ini_state'] = INIT_SORTING
        print 'build sorter'

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
        self.writer_id = self._blkstr.startWriter('SORT_SORT[%s]' % self.mem['tet'])
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
        if self.mem['ini_state'] < INIT_SORTING:
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
#                    print 'bad timesample for event in block: t = % s' % t
#                    t = 0
#                print t + self.mem['block_offset']
                event_lst.append((self.mem['tet'], u, t + self.mem['block_offset'], 0, 0, 0))
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
            #tf=65,
            nc=4,
            # internals
            align_at=10,
            #align_at=20,
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
                (MunkHDF5InitialisationHandler, {}),
                (MunkBOTMHandler, {}),
                (MunkBlockstreamOutputHandler, {}),
            ]
        )
        T.run()

#        print 'testing profile'
#        import cProfile
#        cProfile.runctx("""Tetrode(name='BlockstreamTetrode: % s' % tet, exp=exp, blk=blk, tet=tet, algo='onlineBOTMpy', srate=32000.0, tf=30, nc=4, align_at=10, ini_state=INIT_NOT_STARTED, ini_nspks=1000, ss_buf=350, th_pca=6, th_snr=0.7, th_det=3.5, dtype=sp.float32, debug=debug, handlers=[(MunkBlockstreamInputHandler, {}), (MunkOnlineInitialisationHandler, {}), (MunkBOTMHandler, {}), (MunkBlockstreamOutputHandler, {}), ]).run()""", globals=globals(), locals=locals())

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
