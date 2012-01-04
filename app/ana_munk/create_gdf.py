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


"""produce GDF files for an anylsis from the Munk database"""
__docformat__ = 'restructuredtext'


##--- IMPORTS

import scipy as sp
from database import MunkSession, CON_GNODE
import os
import util


##---CONSTANTS

GDF_SAVE_PATH = '/home/phil/Data/Munk/GDF'
DB = MunkSession(dbconfig=CON_GNODE, verbose=False)
ANA = 511
QUERY = """
SELECT t.trialidx, u.algoid+1, s.sample FROM spike s
JOIN unit u ON (u.id = s.unit)
JOIN analysis a ON (a.id=u.analysis)
JOIN trial t ON (t.id=s.trial)
WHERE a.id = %s%s%s
ORDER BY s.trial, s.sample
"""


##--- FUNCTIONS

def get_gdf_data(id_ana, trials=None, units=None):
    """get spike times and extracted waveforms
    
    id_ana : int
        database id of the analysis
    trl_from_to : tuple or None
        tuple of (start,end) giving the boundaries inside the block with respect
        to trial.trialidx. optional, may be omitted by passing None, returning
        all trial in the analysis
        Default=None
    unit_range : tuple
        tuple of 
    """

    # query analysis
    print 'querying for analysis details'
    q_ana = DB.query("""
    SELECT a.kind, e.name, b.name, t.nr
    FROM analysis a
    JOIN experiment e ON (e.id=a.expid)
    JOIN block b ON (b.id=a.block)
    JOIN tetrode t ON (t.id=a.tetrode)
    WHERE a.id = %d
    """ % id_ana)
    if q_ana == []:
        raise ValueError('analysis does not exist (id=%d)' % id_ana)
    print 'done.'

    # query gdf data
    print 'querying for gdf data'
    trl_constraint = ''
    if trials is not None:
        assert isinstance(trials, tuple)
        assert len(trials) == 2
        trials = map(int, trials)
        trl_constraint = ' AND t.trialidx>=%d AND t.trialidx <=%d' % trials
    unt_constraint = ''
    if units is not None:
        assert isinstance(units, (tuple, list))
        unt_constraint = ' AND u.algoid IN (%s)' % str(list(units))[1:-1]
    q_data = DB.query(QUERY % (id_ana, trl_constraint, unt_constraint))
    print 'done.'
    return q_ana[0], q_data


def write_gdf(query_ana, query_data):
    """write gdf file to directory
    
    :Parameters:
        gdf_dir : str
            path to the directory where the gdfs should be placed. a new dir
            will be created named after the analysis id. user executing the
            script has to be permission to write to the directory.
        query_ana : tuple
            tuple holding (id_ana, kind_ana, exp_ana, block_ana, tet_ana)
        query_data : list of tuple
            list of one tuple per spike as (trialidx, unit_label, sample)
    """

    print 'care for the directory'
    if not os.path.exists(GDF_SAVE_PATH):
        os.mkdir(GDF_SAVE_PATH)
    savedir = os.path.join(GDF_SAVE_PATH, str(id_ana))
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    print 'done.'

    # write gdfs
    print 'writing gdf files'
    f = None
    last_trialidx = -1
    while len(query_data) > 0:
        item = query_data.pop(0)
        if item[0] != last_trialidx:
            if f is not None:
                f.close()
            fname = os.path.join(savedir, '%s%04d.t%d.gdf' % (query_ana[1],
                                                              item[0],
                                                              query_ana[3]))
            f = open(fname, 'w')
            print 'starting to write', fname
            last_trialidx = item[0]
        f.write('%05d\t%d\n' % item[1:])
    f.close()
    print 'done.'


def produce_gdf_set(id_ana, trials=None, units=None):
    res = get_gdf_data(id_ana, trials=trials, units=units)
    write_gdf(*res)
    print 'all done.'


##---MAIN

if __name__ == '__main__':

    # checks and init
    import sys
    if len(sys.argv) != 2:
        raise ValueError('Give exactly one argument: 1) db id of the analysis')
    id_ana = int(sys.argv[1])

    produce_gdf_set(id_ana, units=(9, 6, 5, 8, 3, 1))
