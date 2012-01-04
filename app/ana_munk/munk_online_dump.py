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


"""
Created on 16.06.2011

@author: phil
"""

import scipy as sp
from struct import unpack
from common import build_block_toeplitz_from_xcorrs

def load_cmx_data(fname):
    """load nmdaq cove dumping file
    
    group_nr::uint16
    nc::uint16
    tf::uint16
    timestamp:uint64
    data::float[(nc*nc) * (2*tf-1)]
    """

    # inits
    rval = {}

    #read stuff
    f = open(fname, 'rb')
    f.seek(0, 2)
    end = f.tell()
    f.seek(0)
    bytes_read = 0
    while bytes_read < end:
        gid = unpack('H', f.read(2))[0]
        nc = unpack('H', f.read(2))[0]
        tf = unpack('H', f.read(2))[0]
        ts = unpack('Q', f.read(8))[0]
        len_data = 4 * (2 * tf - 1) * (nc * nc)
        xc = sp.frombuffer(f.read(len_data), dtype=sp.float32)
        xc.shape = (nc * nc, 2 * tf - 1)
        rval[ts] = xc
        bytes_read += len_data + 14

    # return
    return rval

def load_spk_data(fname):
    """load nmdaq sort dumping file
    
    group_nr::uint16
    unit_nr::uint16
    timeval::uint64
    event_type::uint16
    user1::uint16
    user2::uint16
    nc::uint16
    sample_count::uint16
    data[sample_count]::int16
    """

    # inits
    rval = {}

    #read stuff
    f = open(fname, 'rb')
    f.seek(0, 2)
    end = f.tell()
    f.seek(0)
    bytes_read = 0
    while bytes_read < end:
        gid = unpack('H', f.read(2))[0]
        uid = unpack('H', f.read(2))[0]
        tv = unpack('Q', f.read(8))[0]
        et = unpack('H', f.read(2))[0]
        u1 = unpack('H', f.read(2))[0]
        u2 = unpack('H', f.read(2))[0]
        nc = unpack('H', f.read(2))[0]
        ns = unpack('H', f.read(2))[0]
        len_data = 2 * ns * nc
        spk_data = sp.frombuffer(f.read(len_data), dtype=sp.int16)
        bytes_read += len_data + 22

        if uid not in rval:
            rval[uid] = []
        rval[uid].append(spk_data)

    # return
    return rval


##---MAIN

if __name__ == '__main__':

    from plot import P, waveforms

    fname_cmx = 'C:\\SHARE\\#cove_14.cove'
    cmx_data = load_cmx_data(fname_cmx)
    P.matshow(cmx_data[min(cmx_data.keys())])

    fname_spk = 'C:\\SHARE\\#spikes_14.spikes'
    spk_data = load_spk_data(fname_spk)
    waveforms(spk_data,
              samples_per_second=32000.0,
              tf=spk_data.values()[0].shape[0],
              plot_mean=True,
              plot_single_waveforms=True,
              plot_separate=True,
              show=True)
