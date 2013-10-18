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

"""constants for the common package"""
__docformat__ = 'restructuredtext'
__all__ = ['INDEX_DTYPE', 'SI8MAX', 'SI16MAX', 'SI32MAX', 'SI64MAX', 'UI8MAX',
           'UI16MAX', 'UI32MAX', 'UI64MAX', 'VERBOSE', 'log']

## IMPORTS

import logging
import scipy as sp

## PACKAGE-LOGGING

logging.basicConfig(level=logging.DEBUG, format='')
log = logging.getLogger('BOTMpy')

## CONSTANTS

## index type
INDEX_DTYPE = sp.dtype(sp.int64)

## integer max values
SI8MAX = sp.iinfo(sp.int8).max
SI16MAX = sp.iinfo(sp.int16).max
SI32MAX = sp.iinfo(sp.int32).max
SI64MAX = sp.iinfo(sp.int64).max
UI8MAX = sp.iinfo(sp.uint8).max
UI16MAX = sp.iinfo(sp.uint16).max
UI32MAX = sp.iinfo(sp.uint32).max
UI64MAX = sp.iinfo(sp.uint64).max


## CLASSES

class VERBOSE(object):
    """verbosity manager"""

    # default modes
    NONE = 0x0
    PRINT = 0x1

    # augmented modes
    PLOT = 0x10

    ## constructor

    def __init__(self, level=None, has_print=False, has_plot=False):
        """

        :param level: int
        :type level: bitfield as hexadecimal
        :param has_print:
        :type has_print:
        :param has_plot:
        :type has_plot:
        :return:
        :rtype:
        """

        # set level
        if isinstance(level, VERBOSE):
            level = level.level
        self.level = level

        # adjust level
        if self.level is None:
            self.level = self.NONE
        if has_print is True:
            self.level |= self.PRINT
        if has_plot is True:
            self.level |= self.PLOT

    ## interface

    def get_is_verbose(self):
        return self.level > self.NONE

    is_verbose = property(get_is_verbose)

    def get_has_print(self, n=0):
        return self.level >= self.PRINT + n

    has_print = property(get_has_print)

    def get_has_plot(self, n=0):
        return self.level >= self.PLOT + n

    has_plot = property(get_has_plot)

## MAIN

if __name__ == '__main__':
    pass
