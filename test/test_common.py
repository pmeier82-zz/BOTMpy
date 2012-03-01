# tests for all the smaller things in the common package

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

import scipy as sp
from StringIO import StringIO
from spikepy.common.constants import INDEX_DTYPE
from spikepy.common.debug_helpers import *

##---TESTS-alphabetic-by-file

class TestCommon(ut.TestCase):
    pass


class TestCommonConstants(ut.TestCase):
    def testIndexDtype(self):
        self.assertEqual(INDEX_DTYPE, sp.dtype(sp.int64))


class TestCommonDebugHelpers(ut.TestCase):
    def setUp(self):
        self.STD_INFO = StringIO()
        self.STD_ERROR = StringIO()

    def testDebugStreams(self):
        self.assertEqual(INDEX_DTYPE, sp.dtype(sp.int64))

##---MAIN

if __name__ == '__main__':
    ut.main()
