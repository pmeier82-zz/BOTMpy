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


"""produce extended GDF file from an MunkPostrgesDB analysis item"""
__docformat__ = 'restructuredtext'


##--- IMPORTS

import scipy as sp
from database import MunkSession, CON_GNODE
from common import sortrows
import os
import util


##---CONSTANTS

#EGDF_SAVE_PATH = '/home/phil/Dropbox/MunkShare/InVivoLouisJulia/EGDF'
EGDF_SAVE_PATH = '/home/phil/Data/Munk/EGDF'
DB = MunkSession(dbconfig=CON_GNODE)
ANA = 511
CUT = (32, 32)
UNIT_QUERY = """
-- return columns
SELECT
s.sample,
d.data[s.sample-%d:s.sample+%d]

-- based on analysis item
FROM
analysis a

-- spiketrain joins
JOIN unit u ON (u.analysis=a.id)
JOIN spike s ON (s.unit=u.id)
JOIN trial t ON (t.id=s.trial)-- 

--trialdata joins
JOIN channel c ON (c.tetrode=a.tetrode)
JOIN trialdata d ON (d.channel=c.id AND d.trial=t.id)

-- contraints
WHERE
u.id=%d AND
t.id=%d

-- assert ordering
ORDER BY s.sample, c.nr
"""


##--- FUNCTIONS

def write_egdf(fname, egdf_data):
    """write egdf_data"""

    myfmt = tuple(['%05d'] + ['%d'] * (egdf_data.shape[1] - 1))
    with open(fname, 'w') as f:
        sp.savetxt(f, egdf_data, fmt=myfmt, delimiter='\t')


def get_unit_data(id_unt, id_trl, cut=(32, 32)):
    """get spike times and extracted waveforms
    
    id_unt : int
        database id of the unit
    id_trl : int
        database id of the trial
    cut : (int, int)
        cutleft and cutright values as tuple of positive int
    """

    # query unit data from database 
    udata = DB.get_unit_data(id_unt, id_trl)
    if udata['spiketrain'].size == 0:
        raise ValueError('zero length spiketrain')
    if udata['kind'].find('single') == -1:
        raise ValueError('unit is no single unit!')
    ai = udata['algoid']
    st = udata['spiketrain']

    # build dict sample->waveform_concat
    q = DB.query(UNIT_QUERY % (CUT[0], CUT[1], id_unt, id_trl))
    rval = []
    for i in xrange(0, len(q), 4):
        if len(q[i][1]) != sum(CUT) + 1:
            continue
        rval.append(sp.hstack([q[i][0], ai] +
                              [q[i + j][1] for j in xrange(4)]))
    return sp.vstack(rval)

def produce_egdf(id_ana):
    """produce extended gdf file for analysis and trial"""

    # checks

    # inits
    q = DB.query("""
    SELECT
      a.kind,
      a.expid,
      a.block,
      a.tetrode,
      a.trialidxstart,
      a.trialidxend,
      e.name,
      b.name,
      t.nr+1
    FROM analysis a
    JOIN experiment e ON(e.id=a.expid)
    JOIN block b ON (b.id=a.block)
    JOIN tetrode t ON (t.id=a.tetrode)
    WHERE a.id = %d
    """ % id_ana)
    if len(q) == 0:
        raise ValueError('did not find analysis with id: %d' % id_ana)
    ana_info = q[0]
    trial_ids = DB.get_trial_range(ana_info[2],
                                   include_error_trials=False,
                                   limit=ana_info[5])
    if len(trial_ids) == 0:
        raise ValueError('empty trial list')
    ulist = DB.get_units_for_analysis(id_ana)
    if not os.path.exists(EGDF_SAVE_PATH):
        os.mkdir(EGDF_SAVE_PATH)
    savedir = os.path.join(EGDF_SAVE_PATH, str(id_ana))
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    with open(os.path.join(savedir, 'info.txt'), 'w') as f:
        f.write('EXPERIMENT: %s\n' % ana_info[6])
        f.write('BLOCK: %s\n' % ana_info[7])
        f.write('TETRODE: %s\n' % ana_info[8])
        f.write('TRL_IDX_START: %s\n' % ana_info[4])
        f.write('TRL_IDX_END: %s\n' % ana_info[5])
    for id_trl in trial_ids:
        fname = DB.get_fname_for_id(id_trl)
        print 'egdf(%s)' % fname
        egdf_data = []
        # per unit
        for id_unt in ulist:
            try:
                egdf_data.append(get_unit_data(id_unt, id_trl, CUT))
            except:
                continue
        egdf_data = sp.vstack(egdf_data)
        egdf_data = sortrows(egdf_data)
        egdf_data = sp.hstack((egdf_data[:, 1:2], egdf_data[:, 0:1], egdf_data[:, 2:]))
        fname = os.path.join(savedir, fname[:-4] + '.gdf')
        write_egdf(fname, egdf_data)
        print 'done.'

    print 'ALL DONE'


##---MAIN

if __name__ == '__main__':

    # checks and init
    import sys
    if len(sys.argv) != 2:
        raise ValueError('Give exactly one argument: 1) db id of the analysis')
    id_ana = int(sys.argv[1])

    # start produing egdfs
    produce_egdf(id_ana)
