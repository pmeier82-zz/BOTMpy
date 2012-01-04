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


"""generates twoarchives for benchmarking spikesorting algorithms"""
__docformat__ = 'restructuredtext'


##---IMPORTS

from PyQt4 import QtGui
import scipy as N
from tables import openFile
from nsim.data_io import MinimalClient
from os.path import join


##---CLASSES

class BenchmarkClient(MinimalClient):
    """test client"""

    def __init__(self, **kwargs):
        """
        :Keywords:
            arc_dir: str
                path to the folder where the benchmark archive should be saved.
            arc_name : str
                name for the benchmark -> /<arc_dir>/bmark_<arc_name>.h5
        """

        # archive
        self.arc_handle = None
        self.arc_dir = kwargs.pop('arc_dir', '.')
        self.arc_name = kwargs.pop('arc_name', '')
        self.arc_len = kwargs.pop('arc_len', 32000 * 10)
        # buffers
        self.save_signal = []
        self.save_gtruth = {}
        self.cnt = 0
        self.save_wf = {}
        # internals
        super(BenchmarkClient, self).__init__(**kwargs)

    # saving methods

    def save_data(self):
        """save data per position of the movement programm"""
        self.arc_handle = openFile(join(self.arc_dir, 'bmark%s.h5' % self.arc_name), 'w')
        self.save_signal = N.vstack(self.save_signal)
        self.arc_handle.createArray(self.arc_handle.root, 'data', self.save_signal)
        gt_grp = self.arc_handle.createGroup(self.arc_handle.root, 'groundtruth')
        for ident in self.save_gtruth:
            id_grp = self.arc_handle.createGroup(gt_grp, str(ident))
            self.arc_handle.createArray(id_grp, 'train', self.save_gtruth[ident])
            self.arc_handle.createArray(id_grp, 'waveform', self.save_wf[ident])
        self.arc_handle.flush()
        self.arc_handle.close()

    ## internals

    def handle_data(self, signal, noise, gtrth):
        """data chunk handler"""

        # append signal and noise
        self.save_signal.append(signal)
        # append groundtruth
        offset = self.cnt * signal.shape[0]
        self.cnt += 1
        for ident in gtrth:
            if ident not in self.save_gtruth:
                self.save_gtruth[ident] = []
                self.save_wf[ident] = self.chunk.units[ident]['wf_buf'][0]
            for evt in gtrth[ident]:
                self.save_gtruth[ident].append(evt + offset)
        # end check
        if offset > self.arc_len:
            self.save_data()
            print 'closing, done'
            self.close()

        # show data
        self.dataplot.set_data(signal)


##---MAIN

def main(args):

    # 1st argument - server address
    ip_str = 'localhost'
    if len(args) > 1:
        ip_str = args[1]

    # qt application
    app = QtGui.QApplication([])

    # the client instance
    win = BenchmarkClient(
        addr=(ip_str, 31337),
        # archive parameters
        arc_dir='C:\\Users\\phil\\Development\\SpiDAQ\\SpikePy\\app\\ana_munk\\eval',
        arc_name='_eval',
        arc_len=32000 * 60 * 2,
        # length parameters
        cnklen=1.0,
    )
    win.initialize()
    win.show()

    # run event loop
    app.exec_()


if __name__ == '__main__':

    import sys
    main(sys.argv)
