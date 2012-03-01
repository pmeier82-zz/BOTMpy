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


"""amplitude histogram to track time-variant waveforms"""
__docformat__ = 'restructuredtext'
__all__ = ['AmplitudeHistogram']


##---IMPORTS

import scipy as sp
from spikepy.common.ringbuffer import MxRingBuffer


##---CLASSES

class AmplitudeHistogram(object):
    """amplitude histogram calculator"""

    ## constructor

    def __init__(self, ampl_range=(-2.0, 2.0, .1), ampl_noise=(-0.2, 0.2),
                 nchan=1, bin_size=32000, bin_hist=3000, ):
        """
        :type ampl_range: tuple(float,float, float)
        :param ampl_range: tuple of (min value, max value, step value) for the
            amplitude binning.
            Default=(-2.0,2.0)
        :type ampl_noise: tuple(float,float)
        :param ampl_noise: a tuple of min and max values for the noise band
            of the histogram. values from the noise ban will be
            omitted/considered equal zero for the purpose of visualisation
            and statistics.
            Default=(-0.2,0.2)
        :type nchan: int
        :param nchan: number of channels in the data, histograms will be
            tracked independently for each of the channels.
            Default=1
        :type bin_size: int
        :param bin_size: size of one histogram-bin in data samples, only used
            if single-bin histograms are calculated from the data.
            Default=32000
        :type bin_hist: int
        :param bin_hist: length of the history, once 'bin_hist' bins have been
            the accumulated, the oldest window will be forgotten (ringbuffer).
            Default=1000
        """

        # parameters
        self._ampl_range = AmplitudeHistogram.range(*ampl_range)
        self._ampl_noise = None
        if isinstance(ampl_noise, tuple):
            if len(ampl_noise) == 2:
                self._ampl_noise = (self._ampl_range > ampl_noise[0]) *\
                                   (self._ampl_range < ampl_noise[1])
                self._ampl_noise = self._ampl_noise[::-1]
            else:
                raise ValueError('ampl_noise must be a tuple of length 2!')
        if nchan < 1:
            raise ValueError('nchan must be a positive integer!')
        self._nchan = int(nchan)
        if bin_size < 1:
            raise ValueError('bin_size must be a positive integer!')
        self._bin_size = int(bin_size)
        if bin_hist < 1:
            raise ValueError('bin_hist must be a positive integer!')
        self._bin_hist = int(bin_hist)
        self._hist_data = MxRingBuffer(
            capacity=self._bin_hist,
            dimension=(self._nchan, self._ampl_range.size - 1),
            dtype=int)
        self._cur_bin = sp.zeros((self._nchan, self._ampl_range.size - 1))
        self._cur_bin_smpl = 0
        self._cache_good = False
        self._cache = None

    def force_new_bin(self):
        """ force a new bin and finalize the current bin"""

        self._hist_data.append(self._cur_bin)
        self._cur_bin_smpl = 0
        self._cur_bin[:] = 0

    def append_bin(self, bin):
        """append an AmplHistBin instance

        :type bin: ndarray like
        :param bin: the amplHistBin to append
        """

        # checks
        bin_ = sp.asanyarray(bin)
        if bin_.shape != self._cur_bin.shape:
            raise ValueError('shape does not match! expected %s, got %s' %
                             (self._cur_bin.shape, bin_.shape))
        if bin_.sum() == 0:
            print '!!appending zero bin!!'

        # append bin
        self._hist_data.append(bin_)

    def append_data_all(self, data, force=False):
        """append bin(s) calculated from a strip of data

        with this method a histogram of the amplitude distribution of the
        passed data is generated as one observation and appended to the
        current amplitude histogram.

        :type data: ndarray
        :param data: the data to generate the bin(s) to append from
        :type force: bool
        :param force: if True, immediately start a new bin before calculation
        """

        # check data
        data_ = sp.asanyarray(data)
        if data.ndim < 2:
            data_ = sp.atleast_2d(data_)
            if data_.shape[0] < data_.shape[1]:
                data_ = data_.T
        nsmpl, nchan = data_.shape
        if nchan != self._nchan:
            raise ValueError('data has channel count %s, expected %s' %
                             (nchan, self._nchan))

        # generate bin set
        bin_set = [0]
        if self._cur_bin_smpl != 0:
            bin_set.append(self._bin_size - self._cur_bin_smpl)
        while bin_set[-1] < nsmpl:
            bin_set.append(bin_set[-1] + self._bin_size)
        if bin_set[-1] > nsmpl:
            bin_set[-1] = nsmpl

        # process bins
        idx = 1
        while idx < len(bin_set):
            data_bin = data_[bin_set[idx - 1]:bin_set[idx], :]
            for c in xrange(self._nchan):
                self._cur_bin[c] += sp.histogram(data_bin[:, c],
                                                 bins=self._ampl_range)[0]
            self._cur_bin_smpl += data_bin.shape[0]
            if self._cur_bin_smpl == self._bin_size:
                self.append_bin(self._cur_bin)
                self._cur_bin[:] = 0
                self._cur_bin_smpl = 0
            idx += 1

    def append_data_peaks(self, data, force=False):
        """append bin(s) calculated from a strip of data

        with this method the data is first queried for peaks. this should
        reduce the noise/smoothness of the histogram as observed from the
        amplitude distribution of the pure signal.

        :type data: ndarray
        :param data: the data to generate the bin(s) to append from
        :type force: bool
        :param force: if True, immediately start a new bin before calculation
        """

        # check data
        data_ = sp.asanyarray(data)
        if data.ndim < 2:
            data_ = sp.atleast_2d(data_)
            if data_.shape[0] < data_.shape[1]:
                data_ = data_.T
        nsmpl, nchan = data_.shape
        if nchan != self._nchan:
            raise ValueError('data has channel count %s, expected %s' %
                             (nchan, self._nchan))

        # generate bin set
        bin_set = [0]
        if self._cur_bin_smpl != 0:
            bin_set.append(self._bin_size - self._cur_bin_smpl)
        while bin_set[-1] < nsmpl:
            bin_set.append(bin_set[-1] + self._bin_size)
        if bin_set[-1] > nsmpl:
            bin_set[-1] = nsmpl

        # process bins
        idx = 1
        while idx < len(bin_set):
            data_bin = data_[bin_set[idx - 1]:bin_set[idx], :]
            for c in xrange(self._nchan):
                self._cur_bin[c] += sp.histogram(data_bin[:, c],
                                                 bins=self._ampl_range)[0]
            self._cur_bin_smpl += data_bin.shape[0]
            if self._cur_bin_smpl == self._bin_size:
                self.append_bin(self._cur_bin)
                self._cur_bin[:] = 0
                self._cur_bin_smpl = 0
            idx += 1

    def get_channel_hist(self, channel=0, omit_noise=True):
        """yield the current amplitude histogram for the requested channel
        of the data

        :type channel : int
        :param cahnnel: channel of the data
        :type omit_noise: bool
        :param omit_noise: if True, omit value in the noise band
        """

        rval = self._hist_data[:][:, channel, :].T[::-1]
        if omit_noise is True:
            rval[self._ampl_noise, :] = 0
        return rval

    def get_hist(self, omit_noise=True):
        """yield tuple of amplitude histograms for all channels of the data

        :type omit_noise: bool
        :param omit_noise: if True, omit value in the noise band
        """

        return tuple([self.get_channel_hist(c, omit_noise)
                      for c in xrange(self._nchan)])

    def plot_ampl_hist(self):
        """plot the amplitude histogram"""

        raise DeprecationWarning('NSIM DEPENDANCY !!!')
        # TODO: fix after nsim is good again

        from PyQt4 import QtCore, QtGui
        from nsim.gui.plotting import MatShow

        CMAP_PARAMS = (
            QtCore.Qt.white,
            QtCore.Qt.red,
            (0.0001, QtCore.Qt.blue),
            (0.25, QtCore.Qt.cyan),
            (0.5, QtCore.Qt.green),
            (0.75, QtCore.Qt.yellow)            )

        class AmplHistWidget(QtGui.QWidget):
            def __init__(self, parent=None, **kwargs):
                """
                :Parameters:
                    parent : QWidget
                        Qt parent.
                """

                # super for the plotting
                super(AmplHistWidget, self).__init__(parent)

                # channels
                self.nchan = kwargs.get('nchan', 4)

                # setup gui components
                self.lo_ampl_ms = QtGui.QVBoxLayout(self)
                self.ampl_ms = []
                for i in xrange(self.nchan):
                    self.ampl_ms.append(MatShow(parent=self,
                                                cmap=CMAP_PARAMS))
                    self.lo_ampl_ms.addWidget(self.ampl_ms[i])

            def update_data(self, ah_data):
                """update the amplitude histogram

                :Parameters:
                    ah_data : tuple
                        tuple of single channel amplitude histograms
                """

                for c in xrange(self.nchan):
                    self.ampl_ms[c].set_data(sp.log(ah_data[c] + 1.0))

        # start QT loop
        app = QtGui.QApplication([])
        ahw = AmplHistWidget(nchan=4)
        ahw.update_data(self.get_hist())
        ahw.show()
        return app.exec_()

    ## static methods

    @staticmethod
    def range(min_val, max_val, bin_size):
        """range function for the binning of the histogram"""

        # checks
        if min_val > max_val:
            min_val, max_val = max_val, min_val

        # calculate start and stop bins
        min_bin = int(sp.floor_divide(min_val, bin_size))
        max_bin = int(sp.floor_divide(max_val, bin_size) + 1)

        # return
        return sp.arange(min_bin, max_bin) * bin_size


def main1():
    from tables import openFile
    #    from matplotlib import pyplot as P

    #arc = openFile('C:/SVN/Python/SpikePy/posi/test.h5')
    arc = openFile('/home/ff/amplhist.h5')
    ampl = AmplitudeHistogram(ampl_range=(-.5, .2, .01),
                              ampl_noise=(-.05, .05),
                              nchan=4,
                              bin_size=32000,
                              bin_hist=1000)

    for i in xrange(len(arc.listNodes('/'))):
        print 'Processing: Node %s ..' % i,
        ampl.append_data(arc.getNode('/%d' % i, 'data').read())
        print 'done.'

    arc.close()
    del arc
    print 'done. Computing AmplHist...'

    print
    print 'plotting event loop..',
    ampl.plot_ampl_hist()
    print 'done!'

    return ampl


def main2():
    from spikedb import MunkSession

    EXP = 'L011'
    TNR = 7

    DB = MunkSession()
    ampl = AmplitudeHistogram(ampl_range=(-100.0, 100.0, 1.0),
                              ampl_noise=(-8.0, 8.0),
                              nchan=4,
                              bin_size=300000,
                              bin_hist=2500)

    id_exp = DB.get_exp_id(EXP)
    id_blks = [DB.get_block_id(EXP, '%s' % b) for b in
    ['a']]#, 'b', 'c', 'd', 'e']]
    id_tet = DB.get_tetrode_id(id_exp, TNR)
    tlist = []
    for id_blk in id_blks:
        tlist.extend(DB.get_trial_range(id_blk))

    for tid in tlist:
        print 'Processing: %s ..' % DB.get_fname_for_id(tid),
        ampl.append_data_all(DB.get_tetrode_data(tid, id_tet))
        ampl.force_new_bin()
        print 'done.'
    print 'done. Computing AmplHist...'

    DB.close()

    print
    print 'plotting event loop..',
    ampl.plot_ampl_hist()
    print 'done!'

    return ampl

if __name__ == '__main__':
    main2()
