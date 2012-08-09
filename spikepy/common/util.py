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

"""constants for the common package"""
__docformat__ = 'restructuredtext'
__all__ = ['INDEX_DTYPE', 'SI8MAX', 'SI16MAX', 'SI32MAX', 'SI64MAX', 'UI8MAX',
           'UI16MAX', 'UI32MAX', 'UI64MAX', 'deprecated', 'VERBOSE']

##---IMPORTS

import warnings
import scipy as sp

##---CONSTANTS

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

##---DECORATORS

# found here http://code.activestate.com/recipes/577819-deprecated-decorator/
# Author: Giampaolo Rodola <g.rodola [AT] gmail [DOT] com>
# License: MIT
def deprecated(replacement=None):
    """A decorator which can be used to mark functions as deprecated.
    replacement is a callable that will be called with the same args
    as the decorated function.

    >>> @deprecated()
    ... def foo(x):
    ...     return x
    ...
    >>> ret = foo(1)
    DeprecationWarning: foo is deprecated
    >>> ret
    1
    >>>
    >>>
    >>> def newfun(x):
    ...     return 0
    ...
    >>> @deprecated(newfun)
    ... def foo(x):
    ...     return x
    ...
    >>> ret = foo(1)
    DeprecationWarning: foo is deprecated; use newfun instead
    >>> ret
    0
    >>>
    """

    def outer(oldfun):
        def inner(*args, **kwargs):
            msg = "%s is deprecated" % oldfun.__name__
            if replacement is not None:
                msg += "; use %s instead" % (replacement.__name__)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            if replacement is not None:
                return replacement(*args, **kwargs)
            else:
                return oldfun(*args, **kwargs)

        return inner

    return outer

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

    def get_has_print(self):
        return self.level >= self.PRINT

    has_print = property(get_has_print)

    def get_has_plot(self):
        return  self.level >= self.PLOT

    has_plot = property(get_has_plot)

##---MAIN

if __name__ == '__main__':
    pass
