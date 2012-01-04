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
from database import MunkSession, CON_GNODE
from plot import P, waveforms
from common import mcvec_to_conc, mcvec_from_conc, TimeSeriesCovE
from nodes import BOTMNode
import util


##---CONSTANTS

ANA = 433 # L011 A[433], B[434], C[435], D[436], E[437]
DB = MunkSession(dbconfig=CON_GNODE)
DUMMY_TRIAL = DB.query("""SELECT t.id FROM trial t WHERE t.filename = 'DUMMY'""")[0][0]


##--- FUNCTIONS

def load_init(id_ana):
    """load the initialisation data from the database
    
    :Parameters:
        ini_ana : int
            the id of the analsis item to load from
    """

    print 'checking analysis..'
    q = DB.query("""
    SELECT a.kind, a.expid, a.block, a.tetrode, a.trialidxstart, a.trialidxend
    FROM analysis a
    WHERE a.id = %d
    """ % id_ana)
    if q[0][0] != 'INIT':
        raise TypeError('Anlysis(%d) is of kind: %s; required: INIT!' % (id_ana, q[0][0]))
    print 'good.'

    print 'loading analysis data..'
    ulist = DB.get_units_for_analysis(id_ana)
    units = []
    covmx = None
    for id_unit in ulist:
        udata = DB.get_unit_data(id_unit, DUMMY_TRIAL)
        if udata['type'] == 'covmx':
            covmx = udata['waveform']['data']
        else:
            units.append(mcvec_to_conc(udata['data']))
    units = sp.vstack(units)

    return units, covmx, q[0][1], q[0][2], q[0][3], range(q[0][4] - 1, q[0][5])

def show_wf(wf):
    """plots waveforms"""

    wf_data = {}
    for i in xrange(wf.shape[0]):
        wf_data[i] = sp.atleast_2d(wf[i])

    waveforms(wf_data, tf=wf.shape[1] / 4, plot_separate=True, show=False)


##---MAIN

if __name__ == '__main__':

    units, covmx, id_exp, id_blk, id_tet, trange = load_init(ANA)
    tf = units.shape[1] / 4
    nc = 4
    print 'found: tf=%d, nc=%d' % (tf, nc)

#    P.matshow(covmx)
#    show_wf(units)
#    P.show()

    print 'building covariance estimator'
    CE = TimeSeriesCovE(tf=tf, nc=nc)
    CE.update(covmx, direct=True)
    print 'done.'

    print 'building template set'
    temps = sp.zeros((units.shape[0], tf, nc))
    for u in xrange(units.shape[0]):
        temps[u] = mcvec_from_conc(units[u], nc=nc)
    print 'done.'

    print 'building sorting node'
#    templates, 
#    ce=None, 
#    rb_cap=350, 
#    do_adaption=False, 
#    debug=False, 
#    dtype=None
#    # bss node
#    ovlp_taus=[-2, 0, 2], 
#    spk_pr=1e-6, 
#    noi_pr=1e1):
    SS = BOTMNode(temps, ce=CE, debug=True, ovlp_taus=[-9, -6, -3, 0, 3, 6, 9])
    print 'done.'


    trial_ids = DB.get_trial_range(id_blk, limit=10, trange=trange)
    for id_trl in trial_ids:
        trial_name = DB.get_fname_for_id(id_trl)
        print 'sorting %s' % trial_name
        data = DB.get_tetrode_data(id_trl, id_tet)
        SS(data)
#        SS.plot_sorting(show=True)
        SS.sorting2gdf('./gdf/%s.gdf' % trial_name)

    print
    print 'ALL DONE!'
