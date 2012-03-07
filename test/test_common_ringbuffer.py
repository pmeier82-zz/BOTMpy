# tests ringbuffer structures

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from spikepy.common import MxRingBuffer

##---TESTS

class TestMxRingbuffer(ut.TestCase):
    def setUp(self):
        self.rb = MxRingBuffer(6, (4, 4))

    def testInit(self):
        """test initial states"""

        self.assertEqual(self.rb.capacity, 6)
        self.assertEqual(len(self.rb), 0)
        self.assertTupleEqual(self.rb.dimension, (4, 4))

    def testInsertion(self):
        """test for append and extend"""

        self.rb.append(sp.eye(4))
        self.assertEqual(len(self.rb), 1)
        assert_equal(self.rb[0], sp.eye(4))

        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(1, 4)])
        self.assertEqual(len(self.rb), 4)
        assert_equal(self.rb[-1], sp.eye(4) * 4.0)

    def testFill(self):
        """test buffer filling"""

        self.rb.fill(sp.eye(4))
        self.assertEqual(len(self.rb), 6)
        for item in self.rb:
            assert_equal(item, sp.eye(4))

    def testCapacityIncrease(self):
        """test increase of capacity during operation"""

        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(6)])
        self.assertEqual(len(self.rb), 6)
        for i, item in enumerate(self.rb):
            assert_equal(item, sp.eye(4) * (i + 1))
        self.rb.capacity = 10
        self.assertEqual(len(self.rb), 6)
        for i, item in enumerate(self.rb):
            assert_equal(item, sp.eye(4) * (i + 1))

    def testCapacityDecrease(self):
        """test decrease of capacity during operation"""

        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(6)])
        self.assertEqual(len(self.rb), 6)
        for i, item in enumerate(self.rb):
            assert_equal(item, sp.eye(4) * (i + 1))
        self.rb.capacity = 4
        self.assertEqual(len(self.rb), 4)
        for i, item in enumerate(self.rb):
            assert_equal(item, sp.eye(4) * (i + 3))

    def testFlush(self):
        """tes for clear of ringbuffer"""

        self.rb.fill(sp.ones(self.rb.dimension))
        rb = self.rb.flush()
        assert_equal(rb, sp.array([
        sp.ones(self.rb.dimension) for _ in xrange(6)
        ]))
        self.assertEqual(len(self.rb), 0)

    def testMean(self):
        """mean calculation"""

        # empty
        assert_equal(self.rb.mean(), sp.zeros(self.rb.dimension))

        # ones filled
        self.rb.fill(sp.ones(self.rb.dimension))
        assert_equal(self.rb.mean(), sp.ones(self.rb.dimension))

        # eye xrange
        self.rb.clear()
        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(6)])
        assert_equal(self.rb.mean(), sp.eye(4) * 3.5)
        assert_equal(self.rb.mean(2), sp.eye(4) * 5.5)
        assert_equal(self.rb.mean(1), sp.eye(4) * 6.0)

    def testIndexing(self):
        """test for indexing elements and slices"""

        # when empty
        self.assertRaises(IndexError, self.rb.__getitem__, 0)

        # after fill to cap
        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(6)])
        for i in xrange(6):
            assert_equal(self.rb[i], sp.eye(4) * (i + 1))
        assert_equal(self.rb[:2], sp.array([sp.eye(4), sp.eye(4) * 2]))

        # after wrap around
        self.rb.extend([sp.eye(4) * (i + 1) for i in xrange(9)])
        for i in xrange(6):
            assert_equal(self.rb[i], sp.eye(4) * (i + 4))
        assert_equal(self.rb[:2], sp.array([sp.eye(4) * 4, sp.eye(4) * 5]))

if __name__ == '__main__':
    ut.main()
