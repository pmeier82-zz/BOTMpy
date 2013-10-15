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


"""datafile implementation for nas file format"""
__docformat__ = 'restructuredtext'
__all__ = ['NasFile', '_NAS_ROW_HEADER']

##---IMPORTS

import scipy as sp
from .datafile import DataFile

##---CLASSES

class _NAS_ROW_HEADER(object):
    """the metadata preceding a row of data in the NAS file format"""

    FORMAT = '%02d\t%d\t%04d\t%s\t%d\t%04d\t%d\t%s\t%s\t'

    def __init__(self, tetr, unit, trial, time, contact, stereo, pxalign, max,
                 std):
        self.tetr = tetr
        self.unit = unit
        self.trial = trial
        self.time = time
        self.contact = contact
        self.stereo = stereo
        self.pxalign = pxalign
        self.max = max
        self.std = std

    def __str__(self):
        return self.FORMAT % (
            self.tetr,
            self.unit,
            self.trial,
            _NAS_ROW_HEADER.float_to_str(self.time),
            self.contact,
            self.stereo,
            self.pxalign,
            _NAS_ROW_HEADER.float_to_str(self.max),
            _NAS_ROW_HEADER.float_to_str(self.std))

    __call__ = __str__

    @staticmethod
    def float_to_str(val):
        return ('%.3f' % val).replace('.', ',')


class NasFile(DataFile):
    """NAS file format - NeuronMeter"""

    ## constructor

    def __init__(self, filename=None, dtype=None):
        # members
        self.contact_count = None
        self.srate = None
        self.window_before = None
        self.window_after = None

        # super
        super(NasFile, self).__init__(filename=filename, dtype=dtype)

    ## file header handling

    def get_n_data_points(self):
        """sample count per waveform

        :rtype: int
        :returns: sample count per waveform
        """

        return self.srate * (self.window_after + self.window_before) / 1000000

    def write_header(self, contact_count, srate, window_before, window_after):
        """write the NAS file format header

        :type contact_count: int
        :param contact_count: electrodes per recording point
        :type srate: int
        :param srate: sampling rate in Hz
        :type window_before: int
        :param window_before: cut_left in us
        :type window_after: int
        :param window_after: cut_right in us
        """

        self.contact_count = contact_count
        self.srate = srate
        self.window_before = window_before
        self.window_after = window_after

        self.fp.write("""[NeuronMeter ASCII Tetrode Spike Wave Form File v1.0]
ContactCount=%d
SampleRate(Hz)=%d
WindowBefore(us)=%d
WindowAfter(us)=%d

Tetrode\tUnit\tTrial\tTime\tContact\tStereo\tPxAlign\tMax\tStdDev\tCnt1Ampl1
 Cnt1Ampl2 .. Cnt1AmplN\tCnt2Ampl1 Cnt2Ampl2 .. Cnt2AmplN ....
""" % (self.contact_count, self.srate, self.window_before, self.window_after))

    ## row handling

    def write_row(self, tetr=0, unit=0, trial=0, time=0, data=None):
        """write one row of data to the file

        :type tetr: int
        :param tetr: tetrode id
        :type unit: int
        :param unit: unit id
        :type trial: int
        :param trial: trial id
        :type time: float
        :param time: time in us
        :type data: sequence
        :param data: waveform data
        """

        # prepare
        if data is None:
            raise ValueError('data is None')
        data_len = data.size / self.contact_count
        assert data_len == round(
            data.size / self.contact_count), 'undefined data length'
        stacked_data = sp.vstack(
            [data[i * data_len:(i + 1) * data_len] for i in
             xrange(self.contact_count)]).T
        contact = stacked_data.ptp(axis=0).argmax() + 1
        stereo = int(10 ** (
            int(sp.rand() * 4) % 4)) # we normaly distribute across channels
        rowheader = _NAS_ROW_HEADER(
            tetr,
            unit,
            trial,
            time,
            contact,
            stereo, # stereo - unclear how this is derived - may break timing
            0,
            # pxalign - stupid unit of measurement, unclear how this maps to
            # anything
            data.max(),
            # max - unclear how this is derived for multichanel waveforms
            data.std()
            # std - unclear how this is derived for multichannel waveforms
        )

        # write row
        self.fp.write(rowheader())
        for c in xrange(self.contact_count):
            if c > 0:
                self.fp.write('\t')
            for i in xrange(data_len):
                if i > 0:
                    self.fp.write(' ')
                self.fp.write('%d' % data[c * data_len + i])
        self.fp.write('\n')
        self.fp.flush()

    ## datafile interface

    def _close(self):
        self.fp.close()

    def _closed(self):
        return self.fp.closed

    def _filename(self):
        return self.fp.name

    def _initialize_file(self, filename, **kwargs):
        # open file
        self.fp = open(filename, 'w')

    @staticmethod
    def read_data(self, filename):
        conv = {3:lambda s:float(str(s).replace(",", "."))}
        return sp.genfromtxt(filename, skiprows=7, converters=conv)[:, 9:]

    @staticmethod
    def check_nas_file(filename):
        fh = open(filename)
        # To be implemented
        fh.close(filename)

    @staticmethod
    def get_cols(filename, cols):
        conv = {3:lambda s:float(str(s).replace(",", "."))}
        return sp.genfromtxt(filename, skiprows=7, usecols=cols,
                             converters=conv)

    @staticmethod
    def get_timepoints(filename):
        return NasFile.get_cols(filename, (3,))

    @staticmethod
    def get_gdf(filename):
        return NasFile.get_cols(filename, (1, 3,))

if __name__ == '__main__':

#    fname = '\\\\nr05\data_nr05\Felix\write_test.nas'
    fname = './test.nas'
    from botmpy.common.datafile import GdfFile

    print 'Starting test...'
    headers = {}
    data = {}
    nSpikes = 100
    nTetr = 3
    nClus = 3
    #nf = NasFile('\\\\nr05\data_nr05\Felix\write_test.nas')
    nf = NasFile('write_test.nas')
    srate = 32000
    nf.write_header(4, srate, 1000, 1500)
    nP = nf.get_n_data_points()
    print nP
    unit = 0
    trial = 1
    for tetrode in xrange(nTetr):
        gdf_file = '%d_tetrode.gdf' % tetrode
        gdf = {}
        for c in xrange(nClus):
            mean = sp.randn(nP * 4) * 30
            gdf[c] = []
            for s in xrange(nSpikes):
                data = 10 * sp.randn(nP * 4) + mean
                time = 1 + s * 9000 / nSpikes + c * 10
                nf.write_row(tetrode, unit, trial, time,
                             data) # 2, 0000, 0, 6.238, 10.010, data)
                gdf[c].append(round(time * srate / 1000))
        GdfFile.write_gdf(gdf_file, gdf)

    nf.close()
    print 'Done.'
