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
#   * 2013-10-30 pmeier82 :: no tests given, old testing code still present
#_____________________________________________________________________________
#

## IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

import warnings
from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from botmpy.nodes import ArtifactDetectorNode, SpectrumArtifactDetector

## TESTS

class TestArtifactDetectorNode(ut.TestCase):
    def setUp(self):
        pass

    def testMissing(self):
        warnings.warn("no test provided for class ArtifactDetectorNode")


class TestSpectrumArtifactDetectorNode(ut.TestCase):
    def setUp(self):
        pass

    def testMissing(self):
        warnings.warn("no test provided for class SpectrumArtifactDetectorNode")

    ## TESTS-OLD

    """
    from os import listdir, path as osp
    from spikeplot import mcdata, plt
    from botmpy.common import XpdFile
    from botmpy.nodes import SDMteoNode as SDET

    tf = 65
    AD = ArtifactDetectorNode()
    SD = SDET(tf=tf, min_dist=int(tf * 0.5))
    XPDPATH = '/home/phil/Data/Munk/Louis/L011'

    for fname in sorted(filter(lambda x:x.startswith('L011') and
                                        x.endswith('.xpd'),
                               listdir(XPDPATH)))[:20]:
        arc = XpdFile(osp.join(XPDPATH, fname))
        data = arc.get_data(item=7)
        AD(data)
        print AD.events
        print AD.get_nonartefact_epochs()
        print AD.get_fragmentation()
        SD(data)
        f = mcdata(data=data, other=SD.energy, events={0:SD.events},
                   epochs=AD.events, show=False)
        for t in SD.threshold:
            f.axes[-1].axhline(t)
        plt.show()
    """

## MAIN

if __name__ == "__main__":
    ut.main()

## EOF
