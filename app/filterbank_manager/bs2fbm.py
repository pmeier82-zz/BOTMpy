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


import scipy as sp
from fbm_group import FilterGroup
from blockstream import (load_blockstream, BS3Reader, WAVEProtocolHandler,
                         COVEProtocolHandler, Queue, Empty, USE_PROCESS,
                         BS3SortSetupBlock)
from common import get_cut, mcvec_to_conc
import sys
import traceback

##---FUNCTIONS

def wave2spks(waveblock, target_gid):
    s_wfm = []
    s_uid = []
    s_tvl = []
    for wave in waveblock.event_lst:
        gid, uid, tv, nc, ns, wf = wave
        if gid != target_gid:
            continue
        s_wfm.append(wf)
        s_uid.append(uid)
        s_tvl.append(tv)
    return sp.asarray(s_wfm), sp.asarray(s_uid), sp.asarray(s_tvl)


def send_sort_preamble(bslib, wid, query, gid):
    rval = True
    try:
        params, ce, units = query
        ulst = []
        for u in units:
            ulst.append((u,
                         mcvec_to_conc(units[u].f),
                         mcvec_to_conc(units[u].xi),
                         units[u].snr, 1, 0, 0))
        glst = [[gid, params['nc'], params['tf'], get_cut(params['tf'])[0],
                 ce.get_cmx(tf=params['tf'], chan_set=params['cs']), ulst]]

        preamble = BS3SortSetupBlock(glst)
        bslib.setPreamble(wid, preamble.BLOCK_CODE, preamble.payload(),
                          len(preamble))
    except:
        rval = False
        traceback.print_exc()
    finally:
        return rval


def filter_group_manager_intermediate(gid, tf, verbose=False):
    wave_det_reader = None
    wave_fbr_reader = None
    cove_reader = None
    BSLIB = None
    status_counter = 0
    try:
        # filter group
        params = {}
        params['srate'] = 32000.0
        params['tf'] = int(tf)
        params['cut'] = get_cut(params['tf'])
        params['nc'] = 4
        params['cs'] = tuple(range(params['nc']))
        params['train_amount'] = 500
        params['spike_amount'] = 100
        params['spike_timeout'] = int(params['srate'] * 10)
        params['ali_at'] = int(sum(params['cut']) / 6.0)
        params['sigma'] = 3.0
        params['snr_th'] = 0.5
        fbg = FilterGroup(0, params=params, pca_dim=4, verbose=verbose)

        # blockstream
        BSLIB = load_blockstream('FilterBankManager')
        BSLIB.setAppStatus('INITIALIZING', 'yellow')

        # bs readers [det wave, fbr wave, cove]
        Qcove = Queue()
        Qdet = Queue()
        Qfbr = Queue()
        cove_reader = BS3Reader(COVEProtocolHandler, Qcove,
                                ident='fbmCoveR')
        cove_reader.start()
        wave_det_reader = BS3Reader(WAVEProtocolHandler, Qdet,
                                    ident='fbmDetWaveR')
        wave_det_reader.start()
        wave_fbr_reader = BS3Reader(WAVEProtocolHandler, Qfbr,
                                    ident='fbmFbrWaveR')
        wave_fbr_reader.start()

        # bs writer [sort for preambles]
        # int16 startWriter(const char* name,const char* streamType);
        sort_fbc_writer_id = BSLIB.startWriter('fbmFbcSortW', 'SORT')

        # doomsday loop
        while True: # really?
            #print Qdet.qsize()
            try:
                wave_item = Qdet.get(timeout=.1)
                s,u,t = wave2spks(wave_item, gid)
                if len(s) > 0:
                    fbg.update(s,u,t)
                del wave_item
            except Empty:
                pass

            #print Qfrb.qsize()
            try:
                wave_item = Qfbr.get(timeout=.1)
                s,u,t = wave2spks(wave_item, gid)
                if len(s) > 0:
                    fbg.update(s,u,t)
                del wave_item
            except Empty:
                pass

            #print Qcove.qsize()
            if not Qcove.empty(): # assuming little load on that reader

                # check for group

                cove_item = Qcove.get()
                if cove_item.data_lst[0] == gid:
                    fbg.update_ce(cove_item)
                del cove_item

            if fbg.need_promote:
                q = fbg.query()
                if q is not None:
                    preamble_sent = send_sort_preamble(BSLIB,
                                                       sort_fbc_writer_id,
                                                       q,
                                                       gid)
                    if verbose is True:
                        print 'sending sort preamble', preamble_sent

            status_counter += 1
            if status_counter == 5:
                status = {True:'TRAINING', False:'SORTING'}[fbg.is_training]
                spkbuf = 'spks[%d :: %d]' % (len(fbg.spk_buf),
                                             fbg._spikes_rejected)
                if fbg.model:
                    model = 'units[%d]' % len(fbg.unit)
                else:
                    model = 'None'
                BSLIB.setAppStatus('%s\n%s\n%s' % (status, spkbuf, model),
                                   'green')
                status_counter = 0
    except:
        traceback.print_exc()
    finally:
        if BSLIB:
            BSLIB.setAppStatus('SHUTDOWN', 'red')
        if wave_det_reader:
            wave_det_reader.stop()
            if USE_PROCESS is True:
                wave_det_reader.terminate()
        if wave_fbr_reader:
            wave_fbr_reader.stop()
            if USE_PROCESS is True:
                wave_fbr_reader.terminate()
        if cove_reader:
            cove_reader.stop()
            if USE_PROCESS is True:
                cove_reader.terminate()
        if BSLIB:
            BSLIB.finalizeAll()
        print 'exit!'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        gid = int(sys.argv[1])
    else:
        gid = 0
    if len(sys.argv) > 2:
        tf = int(sys.argv[2])
    else:
        tf = 65
    if len(sys.argv) > 3:
        verbose = str(sys.argv[3]).lower() == 'true'
    else:
        verbose = True
    filter_group_manager_intermediate(gid, tf, verbose)
    sys.exit(0)
