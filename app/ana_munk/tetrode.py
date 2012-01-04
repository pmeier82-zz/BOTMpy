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


"""xpd file based NTrode"""
__docformat__ = 'restructuredtext'


##--- IMPORTS

from ntrode import NTrode
import scipy as sp
from time import time
import util


##--- CLASSES

class Tetrode(NTrode):
    """NTrode for sorting Munk in-vivo data"""

    def __init__(
        self,
        # experiment
        name=None,
        exp='L011',
        blk='a',
        tet=1,
        algo='unknown',
        # internals
        tf=47,
        nc=4,
        srate=32000.0,
        dtype=sp.float32,
        # debug
        debug=False,
        # handlers
        handlers=None,
        # keywords
        ** kwargs
    ):
        """initialise a Tetrode"""

        # super
        init_handlers = []
        if handlers is not None:
            init_handlers.extend(handlers)
        super(Tetrode, self).__init__(
            name='T %02d : B %s : E %s' % (tet, blk, exp),
            init_handlers=init_handlers
        )

        # update self.mem
        self.mem.update(
            exp=exp,
            blk=blk,
            tet=tet,
            algo=algo,
            tf=tf,
            nc=nc,
            srate=srate,
            dtype=dtype,
            debug=debug,
            timing={
                'start' : None,
                'end' : None,
                'duration' : None,
            },
            input_idx='Tetrode.__init__',
            name=name or self.name
        )
        for k in kwargs:
            if k not in self.mem:
                self.mem[k] = kwargs[k]

    def _check_cycle_criterion(self):
        return self.mem['input_idx'] is not None

    def _initialise(self):
        self.mem['timing'].update(start=time())

    def _finalise(self):
        self.mem['timing'].update(end=time())
        self.mem['timing']['duration'] = \
            self.mem['timing']['end'] - \
            self.mem['timing']['start']

        print
        print self.mem['timing']
        print


###-- MAIN

if __name__ == '__main__':
    pass
