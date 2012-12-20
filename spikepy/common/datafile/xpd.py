# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012 Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Neural Information Processing Group (NI)
#               School for Electrical Engineering and Computer Science
#               Berlin Institute of Technology
#               MAR 5-6, Marchstr. 23, 10587 Berlin, Germany
#               http://www.ni.tu-berlin.de/
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


"""datafile implementation for xpd file format"""
__docformat__ = 'restructuredtext'
__all__ = ['XpdFile', '_XPD_TH', '_XPD_CH']

##---IMPORTS

import scipy as sp
from struct import Struct
from .datafile import DataFile, DataFileError

##---CONSTANTS

_CONV_I = Struct('I')
_CONV_H = Struct('H')
_CONV_B = Struct('B')
_CONV_d = Struct('d')
_CONV_s = Struct('s')
_CONV_7s = Struct('7s')
_CONV_255s = Struct('255s')

##--- CLASSES

class _XPD_TH(object):
    """XPD trial header struct

        ===== =============================
        Size  XPD Trial Header Member
        ===== =============================
        H     header size
        I     data size
        7s    name of generating program
        B     version identifier = 86 / 'V'
        H     NM build no.
        B     high ver. = 0
        B     low ver. = 12
        I     trial no.
        B     stimulus code
        B     error byte
        4H    time stamp
        255s  comment
        255s  additional comment
        Total 542 bytes
        ===== =============================
    """

    SIZE = 542

    def __init__(self, fp):
        """
        :type fp: file
        :param fp: open file at seek(header_start)
        """

        # read buffer
        buf = fp.read(self.SIZE)

        # extract trial header information
        self.header_size = _CONV_H.unpack(buf[0:2])[0]
        self.data_size = _CONV_I.unpack(buf[2:6])[0]
        self.name = _CONV_7s.unpack(buf[6:13])[0]
        self.version = _CONV_s.unpack(buf[13:14])[0]
        self.nm_build_no = _CONV_H.unpack(buf[14:16])[0]
        self.high_ver = _CONV_B.unpack(buf[16:17])[0]
        self.low_ver = _CONV_B.unpack(buf[17:18])[0]
        self.trial_no = _CONV_I.unpack(buf[18:22])[0]
        self.stimulus = _CONV_B.unpack(buf[22:23])[0]
        self.error = _CONV_B.unpack(buf[23:24])[0]
        self.timestamp = (_CONV_H.unpack(buf[24:26])[0],
                          _CONV_H.unpack(buf[26:28])[0],
                          _CONV_H.unpack(buf[28:30])[0],
                          _CONV_H.unpack(buf[30:32])[0])
        self.comment = _CONV_255s.unpack(buf[32:287])[0]
        self.comment = self.comment[:self.comment.find('\x00')]
        self.add_comment = _CONV_255s.unpack(buf[287:542])[0]
        self.add_comment = self.add_comment[:self.add_comment.find('\x00')]

    def __str__(self):
        rval = self.__repr__()
        rval += '\nheader_size\t%d\n' % self.header_size
        rval += 'data_size\t%d\n' % self.data_size
        rval += 'name\t\t%s\n' % self.name
        rval += 'version\t\t%s\n' % self.version
        rval += 'nm_build_no\t%d\n' % self.nm_build_no
        rval += 'high_ver\t%d\n' % self.high_ver
        rval += 'low_ver\t\t%d\n' % self.low_ver
        rval += 'trial_no\t%d\n' % self.trial_no
        rval += 'stimulus\t%d\n' % self.stimulus
        rval += 'error\t%d\n' % self.error
        rval += 'timestamp\t%s\n' % str(self.timestamp)
        rval += 'comment\t\t%s\n' % self.comment
        rval += 'add_comment\t%s\n' % self.add_comment
        return rval


class _XPD_CH(object):
    """XPD channel header struct

        ===== =====================
        Size  XPD Channel Header
        ===== =====================
        I     channel no.
        d     sample rate in kHz
        d     offset-x of channel
        I     data length in samples
        Total 24 bytes
        ===== =====================
    """

    SIZE = 24

    def __init__(self, fp):
        """
        :type fp: file
        :param fp: open file at seek(header_start)
        """

        # read buffer
        buf = fp.read(self.SIZE)

        # extract channel header information
        self.channel_no = _CONV_I.unpack(buf[0:4])[0]
        self.sample_rate = _CONV_d.unpack(buf[4:12])[0]
        self.x_offset = _CONV_d.unpack(buf[12:20])[0]
        self.n_sample = _CONV_I.unpack(buf[20:24])[0]
        self.data_offset = fp.tell()

    def __str__(self):
        rval = self.__repr__()
        rval += '\nchannel_no\t%d\n' % self.channel_no
        rval += 'sample_rate\t%f\n' % self.sample_rate
        rval += 'x_offset\t%f\n' % self.x_offset
        rval += 'n_sample\t%d\n' % self.n_sample
        rval += 'data_offset\t%d' % self.data_offset
        return rval


class XpdFile(DataFile):
    """XPD file from - Matthias Munk Group @ MPI Tübingen"""

    ## constructor

    def __init__(self, filename=None, dtype=None, cache=False):
        # members
        self.trial_header = None
        self.n_achan = None
        self.n_dchan = None
        self.n_echan = None
        self.max_achan = None
        self.achan_header = None
        self.dchan_header = None
        self.echan_header = None
        self.cache = cache
        self._cache = None

        # super
        super(XpdFile, self).__init__(filename=filename, dtype=dtype)

    def __del__(self):
        super(XpdFile, self).__del__()
        self._cache = None

    ## implementation

    def _initialize_file(self, filename, **kwargs):
        # open file
        self.fp = open(filename, 'rb')

        # trial header
        if _CONV_H.unpack(self.fp.read(2))[0] != 120:
            self.fp.close()
            raise DataFileError('unexpected input while reading trial '
                                'header for file %s' % self.fp.name)
        self.trial_header = _XPD_TH(self.fp)

        # analog channel headers
        if _CONV_H.unpack(self.fp.read(2))[0] != 123:
            self.fp.close()
            raise DataFileError('unexpected input while reading analog '
                                'channel header for file %s' %
                                self.fp.name)
        self.n_achan = _CONV_I.unpack(self.fp.read(4))[0]
        self.achan_header = {}
        self.max_achan = -1
        for _ in xrange(self.n_achan):
            ach = _XPD_CH(self.fp)
            if ach.channel_no > self.max_achan:
                self.max_achan = ach.channel_no
            self.achan_header[ach.channel_no] = ach
            self.fp.seek(ach.n_sample * 2, 1)

        # init the cache for the analog channels
        if self.cache is True:
            self._cache = {}

        # digital channel headers
        if _CONV_H.unpack(self.fp.read(2))[0] != 121:
            self.fp.close()
            raise DataFileError('unexpected input while reading digital '
                                'channel header for file %s' %
                                self.fp.name)
        self.n_dchan = _CONV_I.unpack(self.fp.read(4))[0]
        self.dchan_header = {}
        for _ in xrange(self.n_dchan):
            dch = _XPD_CH(self.fp)
            self.dchan_header[dch.channel_no] = dch
            self.fp.seek(dch.n_sample * 4, 1)

        # event channel headers
        if _CONV_H.unpack(self.fp.read(2))[0] != 122:
            self.fp.close()
            raise DataFileError('unexpected input while reading event '
                                'channel header for file %s' %
                                self.fp.name)
        self.n_echan = _CONV_I.unpack(self.fp.read(4))[0]
        self.echan_header = {}
        for _ in xrange(self.n_echan):
            ech = _XPD_CH(self.fp)
            self.echan_header[ech.channel_no] = ech
            self.fp.seek(ech.n_sample * 4, 1)

        self.fp.seek(0)

    def _close(self):
        self.fp.close()

    def _closed(self):
        return self.fp.closed

    def _filename(self):
        return self.fp.name

    def _get_data(self, **kwargs):
        """returns a numpy array of the data with samples on the rows and
        channels on the columns. channels may be selected via the channels
        parameter.

        get data for one tetrode (default all channels) as ndarray

        :type item: int
        :keyword item: tetrode id. starts at 1!!
            Default = 1
        :type chans: list
        :keyword chans: Channel list.
            Default = [0,1,2,3]
        """

        # keywords
        item = kwargs.get('item', 1)
        chans = kwargs.get('chans', [0, 1, 2, 3])

        # inits
        my_chans = [item + chans[i] * 16 for i in xrange(len(chans))]
        nsample = 0
        for i in xrange(4):
            if my_chans[i] not in self.achan_header:
                continue
            else:
                if self.achan_header[my_chans[i]].n_sample > nsample:
                    nsample = self.achan_header[my_chans[i]].n_sample
        if nsample == 0:
            raise IndexError('no data for tetrode %s' % item)

        # collect data
        rval = sp.zeros((nsample, len(chans)), dtype=self.dtype)
        for i in xrange(len(my_chans)):
            load_item = None

            # check cache
            if self.cache is True:
                if my_chans[i] in self._cache:
                    item = self._cache[my_chans[i]]
                    if not isinstance(item, sp.ndarray):
                        load_item = None
                    else:
                        load_item = item

            # load from file
            if load_item is None:
                try:
                    load_item = self._get_achan(my_chans[i])
                except IndexError:
                    load_item = sp.zeros(nsample, dtype=self.dtype)
                if self.cache is True:
                    self._cache[my_chans[i]] = load_item

            # unfortunately the channel shapes are not always consistent
            # across the tetrode. but we preallocate space such that the
            # largest item may fit in the buffer.
            rval[:load_item.size, i] = load_item

        # return stuff
        if not rval.any():
            raise IndexError('no data for tetrode %s' % item)
        return rval

    ## private helpers

    def _get_achan(self, idx):
        """yields an analog channel as ndarray

        :type idx: int
        :param idx: channel id
        """

        # checks
        if idx >= self.max_achan:
            raise IndexError('not a valid channel: %s' % idx)
        if idx not in self.achan_header:
            raise IndexError('no data for this channel: %s' % idx)

        # get data
        self.fp.seek(self.achan_header[idx].data_offset)
        byte_data = self.fp.read(self.achan_header[idx].n_sample * 2)

        # return
        if len(byte_data) == 0:
            return sp.array([], dtype=sp.int16)
        return sp.frombuffer(byte_data, dtype=sp.int16)

    def _get_echan(self, idx):
        """yields an event channel as ndarray

        :type idx: int
        :param idx: channel id.
        """

        # checks
        if idx not in self.echan_header:
            raise IndexError('no data for this channel: %s' % idx)

        # get data
        self.fp.seek(self.echan_header[idx].data_offset, 0)
        byte_data = self.fp.read(self.echan_header[idx].n_sample * 4)

        # return
        if len(byte_data) == 0:
            return sp.array([], dtype=sp.int32)
        return sp.frombuffer(byte_data, dtype=sp.int32)

    def get_available_tetrodes(self):
        """yields the set of available tetrodes"""

        return [k for k in self.achan_header.keys() if 1 <= k <= 16]

##--- MAIN

if __name__ == '__main__':
    arc = XpdFile('/home/phil/Data/Munk/Louis/L011/L0111001.xpd')
    X = arc.get_data(item=1)
    print X
    del arc, X
