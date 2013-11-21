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

"""detector nodes for multi-channeled data

These detectors find features and feature epochs on multi-channeled signals by
application of a threshold to a transform of the input signal. There are several
kinds of standard detectors implemented already.

To implement a new spike detector node create a new module in this package. The
module has to end with "_detection" to be recognized as an implementation file.
Create a new class subclassing
`botmpy.nodes.spike_detection.base.ThresholdDetectorNode`.
"""
__docformat__ = 'restructuredtext'
__all__ = ['EnergyNotCalculatedError', 'ThresholdDetectorNode']

## PACKAGE

from .threshold_detector import EnergyNotCalculatedError, ThresholdDetectorNode

# import all spike_detection implementations from namespace
import os
import sys

_pkg_path = os.path.dirname(os.path.abspath(__file__))

for _mod_name in [_n for _n, _e in [os.path.splitext(_f) for _f in os.listdir(_pkg_path)]
                  if _n.startswith('detection_') and _e == '.py']:
    _mod = __import__('.'.join([__name__, _mod_name]), fromlist=[_mod_name])
    _mod_cls = [getattr(_mod, _attr) for _attr in dir(_mod) if isinstance(getattr(_mod, _attr), type)]
    for _cls in _mod_cls:
        setattr(sys.modules[__name__], _cls.__name__, _cls)
        __all__.append(_cls.__name__)
# clean up namespace
del os, sys, _pkg_path, _mod_name, _n, _e, _f, _mod, _mod_cls, _attr, _cls

## MAIN

if __name__ == '__main__':
    pass

## EOF
