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


"""matrix based ringbuffer implementation"""
__docformat__ = 'restructuredtext'
__all__ = ['MxRingBuffer']

## IMPORTS

import scipy as sp

## CLASSES

class MxRingBuffer(object):
    """ringbuffer implementation based on pre-allocated ndarray

    Ringbuffer behavior is archived by cycling though the buffer forward,
    wrapping around to the start upon reaching capacity.
    """

    ## constructor

    def __init__(self, capacity=64, dimension=1, dtype=None):
        """
        :type capacity: int
        :param capacity: capacity of the ringbuffer (rows)
            Default=64
        :type dimension: tuple or int
        :param dimension: dimensionality of the items to store in the
            ringbuffer. If int, this will be converted internally to (int,)
            Default=1
        :type dtype: dtype resolvable
        :param dtype: dtype of single entries
            Default=float32
        """

        # checks
        if capacity < 1:
            raise ValueError('capacity < 1')
        if isinstance(dimension, int):
            dimension = (dimension,)
        elif isinstance(dimension, tuple):
            pass
        else:
            raise ValueError('dimension has to be tuple or int')

        # members
        self._capacity = int(capacity)
        self._dimension = dimension
        self._dtype = sp.dtype(dtype or sp.float32)
        self._data = sp.empty((self._capacity,) + self._dimension,
                              dtype=self._dtype)
        self._next = 0
        self._full = False

        # mapping prototypes
        self._idx_belowcap_proto = lambda:range(self._next)
        self._idx_fullcap_proto =\
        lambda:range(self._next, self._capacity) + range(self._next)

        # mappings
        self._idx_append = self._idx_fullcap_proto
        self._idx_retrieve = self._idx_belowcap_proto

    ## properties

    def get_dimension(self):
        return self._dimension

    dimension = property(get_dimension)

    def get_is_full(self):
        return self._full

    is_full = property(get_is_full)

    def get_capacity(self):
        return self._capacity

    def set_capacity(self, value):
        if not isinstance(value, int):
            raise ValueError('takes integer as argument')
        if value < 1:
            raise ValueError('capacity < 1')
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
        """append one datum at the end of the buffer, overwriting the oldest
        datum in the buffer if the capacity has been reached.

        :type datum: ndarray
        :param datum: ndarray of shape :self.dimension:
        """

        # checks
        datum = sp.asarray(datum)
        if datum.shape != self._dimension:
            raise ValueError('datum has wrong dimension! expected %s was %s' %
                             (self._dimension, datum.shape))

        # append
        self._data[self._idx_append()[0], :] = datum

        # index and capacity status bookkeeping
        self._next += 1
        if self._next == self._capacity:
            self._next = 0
            if self._full is False:
                self._idx_retrieve = self._idx_fullcap_proto
                self._full = True

    def extend(self, iterable):
        """append iterable at the end of the buffer using multiple append's

        :type iterable: iterable
        :param iterable: iterable of objects to be stored in the ringbuffer
        """

        for item in iterable:
            self.append(item)

    def tolist(self):
        """return the buffer as a list

        :returns: list- the buffer as a python list
        """

        return self._data.tolist()

    def clear(self):
        """clears the data and resets internals"""

        self._next = 0
        self._full = False
        self._idx_retrieve = self._idx_belowcap_proto
        self._data[:] = 0.0

    def flush(self):
        """return the buffer as a list and clear the RingBuffer

        Convenience method. This returns self.tolist() and calls self.clear()
        afterwards.

        :returns: list - the buffer as a python list of the objects stored
        """

        try:
            return self[:].tolist()
        finally:
            self.clear()

    def mean(self, last=None):
        """yields the mean over the :last: entries

        :type last: int
        :param last: number entries from the back of the ringbuffer to
            include for mean calculation. If None, use all contents
            Default=None
        :returns: ndarray(self.dimension) - mean over the last entries,
            or the appropriate zero element if the ringbuffer is empty.
        """

        # checks
        if len(self) == 0:
            # XXX: changed to just zeros(dim, dtype)
            # return sp.mean(sp.zeros(self._dimension, dtype=self._dtype),
            #                axis=0)
            return sp.zeros(self._dimension, dtype=self._dtype)
        if last is None or last > len(self):
            last = len(self)

        # return
        return sp.mean(self._data[self._idx_retrieve()[-last:], :], axis=0)

    def fill(self, datum):
        """fill all slots of the ringbuffer with the same datum.

        :type datum: ndarray
        :param daaum: ndarray of shape :self.dimension:
        """

        # checks
        datum = sp.asarray(datum)
        if datum.shape != self._dimension:
            raise ValueError('datum has wrong dimension! expected %s was' %
                             (self._dimension, datum.shape))

        # append
        self._data[:] = 1.0
        self._data *= datum

        # index and capacity status bookkeeping
        self._next = 0
        if self._full is False:
            self._idx_retrieve = self._idx_fullcap_proto
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
        return len(self._idx_retrieve())

    def __getitem__(self, k):
        try:
            idx = self._idx_retrieve()[k]
            return self._data[idx, ...]
        except IndexError:
            raise IndexError('ringbuffer index out of range')

    def __iter__(self):
        return self._data[self._idx_retrieve(), ...].__iter__()

## MAIN

if __name__ == '__main__':
    pass
