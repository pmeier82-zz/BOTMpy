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


"""datafile implementation for gdf fileformat"""
__docformat__ = 'restructuredtext'
__all__ = ['GdfFile']

##---IMPORTS

import scipy as sp
from spikepy.common.datafile.datafile import DataFile
from common import sortrows

##---CLASSES

class GdfFile(DataFile):
    """gdf file from Chen Sorter software"""

    ## constuctor

    def __init__(self, filename=None, dtype=sp.float32):
        """
        :Parameters:
            filename : str
                Avalid path to a Wri file on the local filesystem.
            dtype : scipy.dtype
                An object that resolves to a vali scipy.dtype.
        """

        # super
        super(GdfFile, self).__init__(filename=filename, dtype=dtype)

    ## implementation

    def _initialize_file(self, filename, **kwargs):
        self.data = GdfFile.read_gdf(filename)

    @staticmethod
    def read_gdf(file_name):
        """
        reads a .gdf file and stores to a dict

        Seems gdfs can only be read sequentially, so we read in line-wise
        and store
        to lists (one list per unit). After reading we convert each list to
        C{numpy.ndarray}.

        @type file_name: string[path]
        @param file_name: path to the file to read

        @rtype: dict of numpy.ndarray
        @return: one sequence per unit
        """

        rval = {}
        read_file = open(file_name, 'r')

        for line in read_file:
            data = line.strip().split()
            if len(data) != 2:
                continue
            if data[0] not in rval:
                rval[data[0]] = []
            rval[data[0]].append(int(data[1]))
        read_file.close()

        for k in rval.keys():
            rval[k] = sp.array(rval[k])

        return rval

    def _closed(self):
        return False

    @staticmethod
    def write_gdf(filename, gdf):
        """Writes a data file. data can be either a dictionary containing a
        list of timepoints for every neuron
        or
        a dictionary containing ndarrays for every neuron
        or
        a matrix with two columns where the first is the neuron id and the
        second is the timepoint"""
        if isinstance(gdf, dict):
            gdf = GdfFile.convert_dict_to_matrix(gdf)

        sp.savetxt(filename, gdf, '%04d %d')

    @staticmethod
    def convert_dict_to_matrix(gdf_dict):
        timepoints = sp.concatenate(gdf_dict.values())
        gdf = sp.zeros((len(timepoints), 2))
        gdf[:, 0] = timepoints
        idx = 0
        ids = sp.zeros(0)
        for key in sorted(gdf_dict.keys()):
            ids = sp.concatenate((ids, sp.ones((len(gdf_dict[key]))) * idx))
            idx += 1
        gdf[:, 1] = ids
        gdf = sortrows(gdf)[:, [1, 0]]
        return gdf

    @staticmethod
    def convert_matrix_to_dict(gdf_mat):
        gdf = {}
        for i in xrange(gdf_mat.shape[0]):
            if not gdf_mat[i, 0] in gdf.keys():
                gdf[gdf_mat[i, 0]] = []
            gdf[gdf_mat[i, 0]].append(gdf_mat[i, 1])
        return gdf

    def _get_data(self, **kwargs):
        """ Returns the gdf content as a dictionary of ndarrays"""
        return self.data

if __name__ == '__main__':
    import os

    try:
        fname1 = './test1.gdf'
        fname2 = './test2.gdf'
        test_data = [
            '0001 1000',
            '0001 1200',
            '0002 1210',
            '0001 1280',
            '0003 1291',
            '0001 1350',
            '0002 1400',
            ]
        with open(fname1, 'w') as f:
            f.write('\n'.join(test_data))
        g1 = GdfFile(fname1)
        gdf1 = g1.get_data()
        print gdf1
        GdfFile.convert_dict_to_matrix(gdf1)
        GdfFile.write_gdf(fname2, gdf1)
        g2 = GdfFile(fname2)
        gdf2 = g2.get_data()
        print gdf2
        is_same = True
        for i in xrange(len(gdf1.keys())):
            if not sp.allclose(gdf1[sorted(gdf1.keys())[i]],
                               gdf2[sorted(gdf2.keys())[i]]):
                is_same = False
                break
        print 'data is same:', is_same
    finally:
        if os.path.exists(fname1):
            os.remove(fname1)
        if os.path.exists(fname2):
            os.remove(fname2)
