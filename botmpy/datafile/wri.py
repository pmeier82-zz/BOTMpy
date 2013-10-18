# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012-2013, Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Philipp Meier <pmeier82@gmail.com>
#
#               Neural Information Processing Group (NI)
#               School for Electrical Engineering and Computer Science
#               Berlin Institute of Technology
#               MAR 5-6, Marchstr. 23, 10587 Berlin, Germany
#               http://www.ni.tu-berlin.de/
#
# Repository:   https://github.com/pmeier82/BOTMpy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal with the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimers.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimers in the documentation
#   and/or other materials provided with the distribution.
# * Neither the names of Neural Information Processing Group (NI), Berlin
#   Institute of Technology, nor the names of its contributors may be used to
#   endorse or promote products derived from this Software without specific
#   prior written permission.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# WITH THE SOFTWARE.
#_____________________________________________________________________________
#
# Acknowledgements:
#   Philipp Meier <pmeier82@gmail.com>
#_____________________________________________________________________________
#
# Changelog:
#   * <iso-date> <identity> :: <description>
#_____________________________________________________________________________
#


"""datafile implementation for wri file format"""
__docformat__ = 'restructuredtext'
__all__ = ['WriFile', '_WRI_H']

##---IMPORTS

import scipy as sp
from .datafile.datafile import DataFile, DataFileError
from ..funcs_general import dict_list_to_ndarray

##---CONSTANTS

VERBOSE = False

##---CLASSES

class _WRI_H(object):
    """WRI header struct"""

    def __init__(self, fp):
        """
        :type fp: file
        :param fp: open file at seek(0)
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
    """WRI file format - Chen Sorter"""

    ## constructor

    def __init__(self, filename=None, dtype=sp.float32):
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
        """ Returns the wri content as a dictionary of numpy arrays
        :rtype: dict
        :returns: mapping unit id to spike train
        """

        return self.npdata

if __name__ == '__main__':
    w = WriFile('C:\\\\SVN\\\\Datenanalyse\\\\Alle\\\write_test000.wri')
    print w.get_data()
