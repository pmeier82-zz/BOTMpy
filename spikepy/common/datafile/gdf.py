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


"""datafile implementation for gdf file format"""
__docformat__ = 'restructuredtext'
__all__ = ['GdfFile']

##---IMPORTS

import scipy as sp
from .datafile import DataFile
from ..funcs_general import sortrows, dict_list_to_ndarray


##---CLASSES

class GdfFile(DataFile):
    """GDF file format - Chen Sorter"""

    ## implementation

    def _initialize_file(self, filename, **kwargs):
        self.data = GdfFile.read_gdf(filename)

    @staticmethod
    def read_gdf(filename):
        """reads a GDF file and stores to a dict

        :type filename: string[path]
        :param filename: path to the file to read

        :rtype: dict of numpy.ndarray
        :return: one sequence per unit
        """

        rval = {}
        with open(filename, 'r') as f:
            for line in f:
                data = line.strip().split()
                if len(data) != 2:
                    continue
                else:
                    key, sample = map(int, data)
                if key not in rval:
                    rval[key] = []
                rval[key].append(sample)
        return dict_list_to_ndarray(rval)

    def _closed(self):
        return False

    @staticmethod
    def write_gdf(filename, gdf):
        """Writes GDF data file.

        :type filename: str
        :param filename: valid path on the filesystem
        :type gdf: dict or ndarray
        :param gdf: either a dict mapping unit ids to spike trains or a
            ndarray with the keys in the first column and the samples in the
            second column"""

        if isinstance(gdf, dict):
            gdf = GdfFile.convert_dict_to_matrix(gdf)
        sp.savetxt(filename, gdf, fmt='%05d %d')

    @staticmethod
    def convert_dict_to_matrix(gdf):
        """converts a GDF dict representation to a matrix

        :type gdf: dict
        :param gdf: mapping unit ids to spike trains
        :rtype: ndarray
        :returns: first column ids, second column samples
        """

        samples = sp.concatenate(gdf.values())
        rval = sp.zeros((len(samples), 2))
        rval[:, 0] = samples
        idx = 0
        ids = sp.zeros(0)
        for key in sorted(gdf.keys()):
            ids = sp.concatenate((ids, sp.ones((len(gdf[key]))) * idx))
            idx += 1
        rval[:, 1] = ids
        rval = sortrows(rval)[:, [1, 0]]
        return rval

    @staticmethod
    def convert_matrix_to_dict(gdf):
        """converts a GDF ndarray representation to a dict

        :type gdf: ndarray
        :param gdf: first column ids, second column samples
        :rtype: dict
        :returns: mapping unit ids to spike trains
        """

        rval = {}
        for i in xrange(gdf.shape[0]):
            if not gdf[i, 0] in rval.keys():
                rval[gdf[i, 0]] = []
            rval[gdf[i, 0]].append(gdf[i, 1])
        return rval

    def _get_data(self, **kwargs):
        """returns the GDF data as ndarray

        :rtype: dict
        :returns: mapping unit ids to spike trains
        """

        return GdfFile.convert_dict_to_matrix(self.data)

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
