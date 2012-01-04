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
from scipy import linalg as sp_la
from database import CON_GNODE, MunkSession
from common import mcvec_to_conc
from plot import P, waveforms
from nodes import AlignmentNode
import util


##---CONSTANTS

ANA = 435
DB = MunkSession(dbconfig=CON_GNODE)
DUMMY_TRIAL = DB.query("""SELECT t.id FROM trial t WHERE t.filename = 'DUMMY'""")[0][0]
ANGLE = 20


##--- FUNCTIONS

def load_init(id_ana):
    """load the initialisation data from the database
    
    :Parameters:
        ini_ana : int
            the id of the analsis item to load from
    """

    print 'checking analysis..'
    q = DB.query("""
    SELECT a.parameters
    FROM analysis a
    WHERE a.id = %d
    """ % id_ana)
    if q[0][0] != 'INIT':
        raise TypeError('Anlysis(%d) is of type: %s; required: INIT!' % (id_ana, q[0][0]))
    print 'good.'

    print 'loading analysis data..'
    ulist = DB.get_units_for_analysis(id_ana)
    units = []
    covmx = None
    for id_unit in ulist:
        udata = DB.get_unit_data(id_unit, DUMMY_TRIAL)['waveform']
        if udata['type'] == 'covmx':
            covmx = udata['data']
        else:
            units.append(mcvec_to_conc(udata['data']))
    units = sp.vstack(units)

    return units, covmx

def angle_from_vec(a, b):
    """computes angle given two vectors"""

    return 180 * sp.arccos(sp.dot(a, b) / (sp_la.norm(a) * sp_la.norm(b))) / sp.pi

def show_wf(wf):
    """plots waveforms"""

    wf_data = {}
    for i in xrange(wf.shape[0]):
        wf_data[i] = sp.atleast_2d(wf[i])

    waveforms(wf_data, tf=wf.shape[1] / 4, plot_separate=True, show=False)


##---MAIN

if __name__ == '__main__':

    units, covmx = load_init(ANA)
    nunits = units.shape[0]

    P.matshow(covmx)
    show_wf(units)

    print 'alignment of initialisation units'
    AN = AlignmentNode(max_tau=units.shape[1] / 4)
    units_ali = AN(units)
    print 'done.'
    show_wf(units_ali)

    print'checking distinctness of units'
    dis = sp.zeros((nunits, nunits))
    for i in xrange(nunits):
        dis[i] = [angle_from_vec(units[i], units[j]) for j in xrange(nunits)]
    print dis
    P.matshow(dis)
    dis_ali = sp.zeros((nunits, nunits))
    for i in xrange(nunits):
        dis_ali[i] = [angle_from_vec(units_ali[i], units_ali[j]) for j in xrange(nunits)]
    print dis_ali
    P.matshow(dis_ali)

    print 'producing merged units'
    units_idx = sp.ones(nunits, dtype=bool)
    m_units = []
    while units_idx.sum() > 0:
        merge_idx = dis_ali[units_idx.argmax(), units_idx] < ANGLE
        m_units.append(units_ali[units_idx, :][merge_idx].mean(0))
        j = 0
        for i in xrange(nunits):
            if units_idx[i] == True:
                if merge_idx[j] == True:
                    units_idx[i] = False
                j += 1
    m_units = sp.vstack(m_units)
    show_wf(m_units)
    print '%d units surviving out of %d' % (m_units.shape[0], units.shape[0])
    print 'done.'

    P.show()
