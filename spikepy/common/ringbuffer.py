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


"""ring buffer implementation"""
__docformat__ = 'restructuredtext'
__all__ = ['RingBuffer', 'MxRingBuffer']

##---IMPORTS

from collections import deque
import scipy as sp

##---CLASSES

class RingBuffer(object):
    """ringbuffer implementation based on collections.deque

    Ringbuffer behavior is archived by checking for len(data) vs self.capacity
    property, and poping a datum at the opposite side of the buffer if
    capacity
    is reached. collections.deque has O(1) access for pop and popleft.

    This Ringbuffer is prefered for non array-like number types and generic
    python objects. See also MxRingbuffer for a ringbuffer working with on a
    prealocated ndarray.
    """

    ## constructor

    def __init__(self, capacity=64):
        """
        :Parameters:
            capacity : int
                maximum capacity
        """

        self._data = deque()
        self.capacity = capacity

    ## interface methods

    def append(self, datum):
        """append a datum at the end of the buffer, and pop a datum on the
        start
        of the buffer if the capacity is reached

        :Parameters:
            datum : object
                object to store
        """

        self._data.append(datum)
        if len(self._data) > self.capacity:
            self._data.popleft()

    def extend(self, iterable):
        """append iterable at the end of the buffer using multiple appends

        :Parameters:
            iterable : iterable
                iterable of objects to be stored in the ringbuffer

        TODO: may be ineffective if len(iterable) >> capacity
        """

        for item in iterable:
            self.append(item)

    def tolist(self):
        """return the buffer as a python list

        :Returns:
            list
                the buffer as a python list of the objects stored
        """

        return list(self._data)

    def flush(self):
        """return the buffer as a list and clear the RingBuffer

        Convenience method. This returns self.tolist() and calls self.clear()
        afterwards.

        :Returns:
            list
                the buffer as a python list of the objects stored
        """

        try:
            return list(self._data)
        finally:
            self._data.clear()

    ## special methods

    def __str__(self):
        return 'RingBuffer{%s}' % self.capacity

    def __len__(self):
        return len(self._data)

    def __getitem__(self, k):
        return self._data[k]

    def __iter__(self):
        return self._data.__iter__()


class MxRingBuffer(object):
    """ringbuffer implementation based on prealocated ndarray

    Ringbuffer behavior is archived by cycling though the buffer forward,
    wrapping around to the start upon reaching capacity.
    """

    ## constructor

    def __init__(self, capacity=64, dimension=1, dtype=None):
        """
        :Parameters:
            capacity : int
                capacity of the ringbuffer (rows)
            dimension : tuple or int
                dimensionality of the items to store in the ringbuffer. If
                int,
                this will be converted internally to (int,)
            dtype : scipy.dtype resolvable
        """

        # checks
        if capacity < 1:
            raise ValueError('capacity has to be > 0')
        if isinstance(dimension, int):
            dimension = (dimension,)

        # members
        self._capacity = int(capacity)
        self._dimension = dimension
        self._dtype = sp.dtype(dtype or sp.float32)
        self._data = sp.zeros((self._capacity,) + self._dimension,
                              dtype=self._dtype)
        self._next = 0
        self._full = False

        # mapping prototypes
        self._idx_belowcap_proto = lambda:range(self._next)
        self._idx_fullcap_proto =\
        lambda:range(self._next, self._capacity) + range(self._next)

        # mappings
        self._idx_append = self._idx_fullcap_proto
        self._idx_retreive = self._idx_belowcap_proto

    ## properties

    def get_dimension(self):
        return self._dimension

    dimension = property(get_dimension)

    def is_full(self):
        return self._full

    full = property(is_full)

    def get_capacity(self):
        return self._capacity

    def set_capacity(self, value):
        if not isinstance(value, int):
            raise ValueError('takes integer as argument')
        if value < 1:
            raise ValueError('capacity has to be > 0')
        hist = min(len(self), value)
        old_data = self[-hist:].copy()
        self._capacity = value
        self._data = sp.zeros((self._capacity,) + self._dimension,
                              dtype=self._dtype)
        self.clear()
        self.extend(old_data[:hist])

    capacity = property(get_capacity, set_capacity)

    ## methods interface

    def append(self, datum):
        """append one datum at the end of the buffer, overwriting the oldes
        datum in the buffer if the capacity is reached.

        :Parameters:
            datum : ndarray
                ndarray od shape self.dimension
        """

        # checks
        if datum.shape != self._dimension:
            raise ValueError('datum has wrong dimension!')

        # append
        self._data[self._idx_append()[0], :] = datum

        # index and capacity status bookkeeping
        self._next += 1
        if self._next == self._capacity:
            self._next = 0
            if self._full is False:
                self._idx_retreive = self._idx_fullcap_proto
                self._full = True

    def extend(self, iterable):
        """append iterable at the end of the buffer using multiple append's

        :Parameters:
            iterable : iterable
                iterable of objects to be stored in the ringbuffer

        TODO: may be ineffective if len(iterable) >> capacity
        """

        for item in iterable:
            self.append(item)

    def tolist(self):
        """return the buffer as a list

        :Returns:
            list
                the buffer as a python list
        """

        return self._data.tolist()

    def clear(self):
        """clears the data and resets internals"""

        self._next = 0
        self._full = False
        self._idx_retreive = self._idx_belowcap_proto
        self._data[:] = 0.0

    def flush(self):
        """return the buffer as a list and clear the RingBuffer

        Convenience method. This returns self.tolist() and calls self.clear()
        afterwards.

        :Returns:
            list
                the buffer as a python list of the objects stored
        """

        try:
            return self._data.tolist()
        finally:
            self.clear()

    def mean(self, last=None):
        """yields the mean over the last entries

        :Parameters:
            last : int
                the last entries to include for the calculation of the mean.
                 If
                not given or None, include everything.
        :Returns:
            float
                mean over the last entries, or the appropriate zero element if
                the ringbuffer is empty.
        """

        # checks
        if len(self) == 0:
            return sp.mean(sp.zeros(self._dimension, dtype=self._dtype),
                           axis=0)
            # TODO: should we raise an exception here?!
        if last is None or last > len(self):
            last = len(self)

        # return
        return sp.mean(self._data[self._idx_retreive()[-last:], :], axis=0)

    def fill(self, datum):
        """fill all slots of the ringbuffer with the same datum.

        :Parameters:
            datum : ndarray
                ndarray od shape self.dimension
        """

        # checks
        if datum.shape != self._dimension:
            raise ValueError('datum has wrong dimension!')

        # append
        self._data = sp.ones_like(self._data)
        self._data *= datum

        # index and capacity status bookkeeping
        self._next = 0
        if self._full is False:
            self._idx_retreive = self._idx_fullcap_proto
            self._full = True

    ## special methods

    def __str__(self):
        nitems = self._capacity
        if self._full is False:
            nitems = self._next
        return 'MxRingbuffer{items:%s - cap:%s@%s}' % (nitems,
                                                       self._capacity,
                                                       str(self._dimension))

    def __len__(self):
        return len(self._idx_retreive())

    def __getitem__(self, k):
        # checks
        if k > self._data.shape[0]:
            raise KeyError('MxRingbuffer: invalid index')

        # return
        return self._data[self._idx_retreive()[k], :]

    def __getslice__(self, k_start, k_end):
        return self._data[self._idx_retreive()[k_start:k_end], ...]

    def __iter__(self):
        return self._data.__iter__()

##---MAIN

if __name__ == '__main__':
    RB = MxRingBuffer(6, (4, 4))
    print RB
    print
    print 'inserting eye(4) * [0,1,2,3]'
    RB.extend([sp.eye(4) * (i + 1) for i in xrange(4)])
    print RB
    print
    print 'RB[-1:]:'
    print RB[-1:]
    print
    print 'RB[-2:]'
    print RB[-2:]
    print
    print 'RB[-3:]'
    print RB[-3:]
    print
    print 'inserting eye(4) * [0,1,2,3]'
    RB.extend([sp.eye(4) * (i + 1) for i in xrange(4, 8)])
    print RB
    print
    print 'RB[-1:]:'
    print RB[-1:]
    print
    print 'RB[-2:]'
    print RB[-2:]
    print
    print 'RB[-3:]'
    print RB[-3:]
    print
    print 'resizing to cap 4'
    RB.set_capacity(4)
    print RB
    print RB[:]
    print
    print 'filling with aranges'
    xi = sp.array([sp.arange(4) + 1] * 4).T
    RB.fill(xi)
    print RB
    print RB[:]
    print
    print 'mean after fill'
    print RB.mean()
