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


"""gnode data gernation for fake MEA data"""
__docformat__ = 'restructuredtext'


##---IMPORTS

from common import mcvec_from_conc
from database import MunkPostgresSession, CON_GNODE
from plot import waveforms
import scipy as sp


##--CONSTANTS

DB = MunkPostgresSession(dbconfig=CON_GNODE)
DB.connect()
UNITID = 615, 620


##---FUNCTIONS

def generate_data(ns=100000, spike_delta=500):
    """generate data like from a 8x8 channel multi-electrode-array
    
    using tetrode waveforms of two distinct units and a white noise distribution
    """

    # load waveforms
    q = DB.query("""
    SELECT
      waveform.data
    FROM
      public.waveform,
      public.template, 
      public.unit
    WHERE
      waveform.id = template.waveform AND
      unit.id = template.unit AND
      template.trial = 60 AND
      unit.id IN %s
    """ % str(UNITID))
    wf = [mcvec_from_conc(sp.asarray(item[0]), nc=4) for item in q]
    tf = wf[0].shape[0]
    # build MEA array base noise
    rval = sp.random.multivariate_normal(sp.zeros(32), sp.eye(32), ns)
    # build upper tetrode
    idx = spike_delta
    while idx < ns - tf:
        rval[idx:idx + tf, :4] += wf[0]
        idx += spike_delta
    # build lower tetrode
    idx = spike_delta + 200
    while idx < ns - tf:
        rval[idx:idx + tf, -4:] += wf[1]
        idx += spike_delta

    # return
    return rval


def spike_detection(data):
    """find templates and noise covariance matrix"""

    # init
    ns, nc = data.shape

    # covariance


##---MAIN

if __name__ == '__main__':

    # plot imports
    from plot import P

    # generate
    fig = P.figure()
    ax = fig.add_subplot(111)
    X = generate_data(ns=25000)
    nc = X.shape[1]
    delta = 25

    for c in xrange(nc):
        ax.plot(X[:, c] + c * delta)
    R0 = sp.cov(X.T)
    for i in xrange(R0.shape[0]):
        print i, sp.dot(R0[0], R0[i])

    P.matshow(R0)

    # plot stuff
    P.show()
