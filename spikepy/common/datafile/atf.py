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


"""datafile implementation for atf file format"""
__docformat__ = 'restructuredtext'
__all__ = ['AtfFile', '_ATF_H']

##---IMPORTS

import scipy as sp
from .datafile import DataFile, DataFileError

##---CLASSES

class _ATF_H(object):
    """ATF file header"""

    def __init__(self, fp):
        """
        :type fp: file
        :param fp: open file at seek(0)
        """

        # version
        self.version = fp.readline().strip('\'\"\r\n').split()
        if self.version != ['ATF', '1.0']:
            raise DataFileError('wrong version: %s' % self.version)

        # data set structure
        self.datasets = fp.readline().strip('\'\"\r\n').split()
        self.datasets = map(int, self.datasets)
        if len(self.datasets) != 2:
            raise DataFileError('invalid file structure: %s' %
                                str(self.datasets))

        self.signals_exported = None
        self.sweep_times = None
        self.dbconfig = {}

        # signal names
        for _ in xrange(self.datasets[0]):
            line = fp.readline().strip('\'\"\r\n')
            if line.startswith('SignalsExported'):
                self.signals_exported = line.split('=')[-1].split(',')
            elif line.startswith('SweepStartTimesMS'):
                self.sweep_times = sp.fromstring(line.split('=')[1], sep=',')
            else:
                # TODO: if we need other header infos, read in here
                pass

        if self.signals_exported is None:
            raise DataFileError('could not get signal count and names!')

        # column headers
        self.col_headers = fp.readline().strip('\r\n').split('\t')[1:]
        self.col_headers =\
        map(str.strip, self.col_headers, ['\'\"'] * len(self.col_headers))


class AtfFile(DataFile):
    """ATF file format - GenePix software"""

    ## constructor

    def __init__(self, filename=None, dtype=None):
        # members
        self.header = None
        self.nchan = None
        self.ndata = None
        self._sample_times = None
        self._data = None

        # super
        super(AtfFile, self).__init__(filename=filename, dtype=dtype)

    def __del__(self):
        super(AtfFile, self).__del__()
        self.fp, self._data = None, None

    ## implementation

    def _initialize_file(self, filename, **kwargs):
        # open file
        self.fp = open(filename, 'r')

        # read header dbconfig
        self.header = _ATF_H(self.fp)
        self.nchan = len(self.header.signals_exported)
        self.ndata = self.header.datasets[1] - 1 / self.nchan

        # read data
        data = sp.fromstring(self.fp.read(), dtype=self.dtype, sep='\t')
        data.shape = (
            data.shape[0] / self.header.datasets[1],
            self.header.datasets[1]
            )
        data = data.T
        self._sample_times = data[0, :]
        self._data = data[1:, :]
        del data

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

        :type mode: str
        :keyword mode: One of 'all', 'chan', 'data' or 'device'. 'all' return
            all the data in the file. 'chan' returns a specific channel.'data'
            returns. 'device' returns all data for a specific device.
        :type item: int
        :keyword item: identifier as per mode
        :rtype: ndarray
        :returns: requested data
        """

        # checks
        mode = kwargs.get('mode', 'all')
        if  mode not in ['all', 'chan', 'data', 'device']:
            raise DataFileError('unknown mode: %s' % mode)
        item = kwargs.get('item', 0)
        if mode in ['chan', 'data']:
            if item not in range(self.nchan):
                raise DataFileError('mode is %s, unknown item: %s' %
                                    (mode, item))
        elif mode in ['device']:
            if item not in range(2):
                raise DataFileError('mode is %s, unknown item: %s' %
                                    (mode, item))

        # return data copies
        rval = None
        if mode is 'all':
            rval = {}
            for chan in xrange(self.nchan):
                rval[chan] = self._data[chan::self.nchan, :].copy()
        elif mode is 'chan':
            rval = self._data[item::self.nchan, :].copy()
        elif mode is 'data':
            rval = self._data[
                   item * self.nchan:(item + 1) * self.nchan, :].copy()
        elif mode is 'device':
            dev_range = {0:[0, 1], 1:[2, 3, 4, 5]}[item]
            rval = sp.vstack([sp.hstack(self._data[i::self.nchan, :])
                              for i in dev_range]).T.copy()
        return rval

if __name__ == '__main__':
    pass
