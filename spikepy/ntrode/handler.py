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


"""ntrode handler interface

The NTrodeHandler is a compute context container. NTrodes manage instances of
NTrodeHandler and apply their code on the current data. A NTrodeHandler in
this is an sense, abstractly models an algorithm that can be applied to batch
or online data.

The NTrode's memory (NTrode.mem) is used in a read only manner to save
intermediate results from iteration to iteration. NTrodeHandler's bring a
common interface for 1) attaching to a NTrode instance, 2) initialisation of
any members and NTrode-global variables within the namespace of the NTrode
instance it is attached to, and 3) NTrode-state sensitive invocation.
"""

__docformat__ = 'restructuredtext'
__all__ = ['NTrodeHandler']

##---IMPORTS

from .ntrode import NTrode, NTrodeError

##---CLASSES

class NTrodeHandler(object):
    """abstract handler class for use with the NTrode class

    Subclass this abstract handler class and write your code into a method,
    then register your new method by inserting a mapping into the
    self.invoke_for_state dictionary, using the desired NTrode-state as key
    """

    ## constructor

    def __init__(self, ntrode=None):
        """
        :Parameters:
            ntrode : NTrode
                parent reference
        """

        self.is_attached = False
        self.is_initialised = False
        self.mem = None
        self.invoke_for_state = {}

        if ntrode is not None:
            self.attach(ntrode)

    ## publics methods

    def attach(self, ntrode):
        """attach this NTrodeHandler to an NTrode instance"""

        # checks
        if not issubclass(ntrode.__class__, NTrode):
            raise NTrodeError('No NTrode instance given, got %s instead'
            % ntrode.__class__.__name__)
        if self.is_attached is True:
            raise NTrodeError('handler is already attached')
            # grab a ref to the namespace
        self.mem = ntrode.mem
        self.is_attached = True

    def initialise(self):
        """public initialise hook"""

        if self.is_initialised is True:
            return
        self._initialise()
        self.is_initialised = True

    def finalise(self):
        """public finalise hook"""

        self._finalise()
        self.is_attached = False
        self.is_initialised = False
        self.mem = None

    def __call__(self, ntrode_state):
        """invoke this handlers context"""

        if self.is_attached is True and self.is_initialised is True:
            if ntrode_state in self.invoke_for_state:
                self.invoke_for_state[ntrode_state]()

    ## private methods

    def _initialise(self):
        """abstract initialise hook, does nothing"""
        pass

    def _finalise(self):
        """abstract finalise hook, does nothing"""
        pass

##---MAIN

if __name__ == '__main__':
    pass
