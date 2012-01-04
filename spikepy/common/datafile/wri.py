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


"""datafile implementation for wri fileformat"""
__docformat__ = 'restructuredtext'
__all__ = ['WriFile', '_WRI_H']

##---IMPORTS

import scipy as sp
from spikepy.common import dict_list_to_ndarray
from spikepy.common.datafile.datafile import DataFile, DataFileError

##---CONSTANTS

VERBOSE = False

##---CLASSES

class _WRI_H(object):
    """data structure holding information about a recording"""

    def __init__(self, fp):
        """
        :Parameters:
            fp : filepointer
                A file pointer at seek(0)
        """

        # version
        self.srate = fp.readline().strip('\r\n').split()
        if self.srate[0] != 'Sampling':
            raise DataFileError('expected "Sampling:" in first row!"')
        self.srate = int(self.srate[1][10:])
        if VERBOSE:
            print self.srate
            print "header done."


class WriFile(DataFile):
    """wri file from Chen Sorter software"""

    ## constuctor

    def __init__(self, filename=None, dtype=sp.float32):
        """
        :Parameters:
            filename : str
                Avalid path to a Wri file on the local filesystem.
            dtype : scipy.dtype
                An object that resolves to a vali scipy.dtype.
        """

        # members
        self.header = None
        self.data = None
        self.npdata = None

        # super
        super(WriFile, self).__init__(filename=filename, dtype=dtype)

    ## implementation

    def _initialize_file(self, filename, **kwargs):
        # open file
        self.fp = open(filename, 'r')

        # read header info
        self.header = _WRI_H(self.fp)
        self.data = {}
        # read data
        current_unit = None
        line = self.fp.readline().strip('\r\n')
        while line:
            if line.isdigit():
                # we found a spike for the current unit.
                # Current unit should not be None at this point
                self.data[current_unit].append(int(line))
            else:
                # This is a subheader indicating a new unit
                current_unit = line[5]
                self.data[current_unit] = []

            line = self.fp.readline().strip('\r\n')

        # Convert the lists to numpyarrays for the spike train alignment
        # function
        self.npdata = dict_list_to_ndarray(self.data)
        if VERBOSE:
            print "found_units: "
            print self.data.keys()

    def _close(self):
        self.fp.close()

    def _closed(self):
        return self.fp.closed

    def _filename(self):
        return self.fp.name

    def _get_data(self, **kwargs):
        """ Returns the wri content as a dictionary of numpy arrays"""
        return self.npdata

if __name__ == '__main__':
    w = WriFile('C:\\\\SVN\\\\Datenanalyse\\\\Alle\\\write_test000.wri')
    print w.get_data()
