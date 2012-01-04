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
from blockstream import (load_blockstream, Queue, BS3BxpdSetupBlock,
                         BS3BxpdDataBlock)
import traceback

##---CONSTANTS

GID = 0
SRATE = 10000.0

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


def send_sort_preamble(bslib, wid, query):
    rval = True
    try:
        params, ce, units = query
        ulst = []
        for u in units:
            ulst.append((u,
                         mcvec_to_conc(units[u].f),
                         mcvec_to_conc(units[u].xi),
                         units[u].snr, 1, 0, 0))
        glst = [[GID, params['nc'], params['tf'], get_cut(params['tf'])[0],
                 ce.get_cmx(tf=params['tf'], chan_set=params['cs']), ulst]]

        preamble = BS3SortSetupBlock(glst)
        bslib.setPreamble(wid, preamble.BLOCK_CODE, preamble.payload(),
                          len(preamble))
    except:
        rval = False
        traceback.print_exc()
    finally:
        return rval


def run_signal_gen(verbose=False):
    try:
        # blockstream
        BSLIB = load_blockstream('BXPD TriangleSpam')
        BSLIB.setAppStatus('INITIALIZING', 'yellow')

        # bs writer [sort for preambles, and bxpd for triangles]
        # int16 startWriter(const char* name,const char* streamType);
        sort_fbc_writer_id = BSLIB.startWriter('fbmFbcSortW', 'SORT')

        # doomsday loop
        while True: # really?
            #print Qdet.qsize()
            try:
                wave_item = Qdet.get(block=True, timeout=.1)
                fbg.update(*wave2spks(wave_item, GID))
                del wave_item
            except Empty:
                pass

            #print Qfrb.qsize()
            try:
                wave_item = Qfbr.get(block=True, timeout=.1)
                fbg.update(*wave2spks(wave_item, GID))
                del wave_item
            except Empty:
                pass

            #print Qcove.qsize()
            if not Qcove.empty(): # assuming little load on that reader
                cove_item = Qcove.get()
                fbg.update_ce(cove_item)
                del cove_item

            if fbg.need_promote:
                BSLIB.setAppStatus('SORTING [%s]' % len(fbg.unit), 'green')
                q = fbg.query()
                if q is not None:
                    preamble_sent = send_sort_preamble(BSLIB,
                                                       sort_fbc_writer_id,
                                                       q)
                    if verbose is True:
                        print 'sending sort preamble', preamble_sent
    except:
        traceback.print_exc()
    finally:
        BSLIB.setAppStatus('SHUTDOWN', 'red')
        #raw_input()
        wave_det_reader.stop()
        wave_fbr_reader.stop()
        cove_reader.stop()
        if USE_PROCESS is True:
            wave_det_reader.terminate()
            wave_fbr_reader.terminate()
            cove_reader.terminate()
        BSLIB.finalizeAll()
        # TODO: finalize app
        print 'exit!'

if __name__ == '__main__':
    test_fbg(True)
