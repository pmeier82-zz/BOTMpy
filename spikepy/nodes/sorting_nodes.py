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


"""abstract sorting node interface

All sorting algorithm implementations should derive from this interface. The
interface assumes to operate on multichanneled input data that is presented as
numpy.ndarray with samples in the rows and channels in the columns.

The sorter provides for the structure and abstract methods to implement generic
sorting algorithms. To account for a range of different sorting algorithms, as
less assumption as possible are made to the data flow during sorting. The
operation mode of the sorting is assumed to be online in the sense that
arbitrarily small batches of data can be sorted in a sorting step. The sorting
results are presented in a python dictionary with one entry per identified unit,
holding the spiketrain in samples relative to the start of the current data
batch.
"""

__docformat__ = 'restructuredtext'
__all__ = ['SortingNode']

##---IMPORTS

import scipy as sp
from .base_nodes import ResetNode

##---CLASSES

class SortingNode(ResetNode):
    """abstract base class for spike sorting algorithms

    This node encapsulates the process of spike sorting, providing a common
    interface for such algorithms and their results. Individual algorithms
    should be implemented as a subclass of SortingNode.

    The interface assumes to get data (usually multichanneled timeseries) and a
    set of parameters. It should return a python dictionary, with one entry per
    unit, where the entry is a python dictionary as well. The entry should hold
    the spiketrain of that unit and any parameters learned or updated from the
    spike sorting process. The results should be saved under the self.fout
    member variable, which should hold the sorting of the last data that was
    passed to the node or None in case the node has not seen any data yet.

    Also it should be possible to merge the sorted spike trains for GDF export
    and produce an epochs representation for any part of the sorting and for
    the noise.
    """

    def __init__(self, input_dim=None, output_dim=None, dtype=sp.float32):
        """
        :Parameters:
            see mdp.Node
        """

        # super
        super(SortingNode, self).__init__(input_dim=input_dim,
            output_dim=output_dim,
            dtype=dtype)

        # interface members
        self.rval = {}

    ## privates methods

    def _sorting(self, x, *args, **kwargs):
        """to be implemented in subclass"""

        raise NotImplementedError

    ## node implementation

    def is_invertible(self):
        return False

    def is_trainable(self):
        return True

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _execute(self, x, *args, **kwargs):
        """implementation of the spike sorting data flow

        The real sorting is conducted in the self._sorting method. This method
        will just call self._sorting and return the unchanged input.
        """

        self._sorting(x, *args, **kwargs)
        return x


##---MAIN

if __name__ == '__main__':
    pass
