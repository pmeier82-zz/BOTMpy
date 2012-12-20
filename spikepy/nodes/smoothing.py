# -*- coding: utf-8 -*-
#_____________________________________________________________________________
#
# Copyright (c) 2012 Berlin Institute of Technology
# All rights reserved.
#
# Developed by:	Philipp Meier <pmeier82@gmail.com>
#               Neural Information Processing Group (NI)
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

"""smoothing algorithms for multichanneled data"""

__docformat__ = 'restructuredtext'
__all__ = ['SmoothingNode', 'smooth']

##---IMPORTS

import scipy as N
from .base_nodes import ResetNode

##---CONSTANTS

FILTER_KERNELS = {
    5:N.array([-3, 12, 17, 12, -3]),
    7:N.array([-2, 3, 6, 7, 6, 3, -2]),
    9:N.array([-21, 14, 39, 54, 59, 54, 39, 14, -21]),
    11:N.array([-36, 9, 44, 69, 84, 89, 84, 69, 44, 9, -36])
}

##---CLASSES

class SmoothingNode(ResetNode):
    """smooths the data using a gauss kernel of size 5 to 11"""

    ## constructor

    def __init__(self, size=5, input_dim=None, dtype=None):
        """
        :Parameters:
            size : int
                window size for the smoothing window
        """

        # super
        super(SmoothingNode, self).__init__(input_dim=input_dim, dtype=dtype)

        if size in FILTER_KERNELS:
            self.kernel = FILTER_KERNELS[size]
        else:
            raise ValueError('window must be in %s' % FILTER_KERNELS.keys())

    ## node implementation

    def is_invertable(self):
        return False

    def is_trainable(self):
        return False

    def _execute(self, x):
        rval = N.zeros_like(x)
        for c in xrange(x.shape[1]):
            rval[:, c] = N.convolve(x[:, c], self.kernel, 'same')
        return rval / self.kernel.sum()

##---FUNCTIONS

def _basic_smooth(signal, kernel):
    """basic smoothing using the explicit kernel given

    :Parameters:
        signal : ndarray
            multichanneled signal [data,channel]
        kernel : ndarray
            kernel used for smoothing
    """

    if kernel.size >= signal.shape[0]:
        raise ValueError('kernel window size larger than signal length')
    rval = N.zeros_like(signal)
    for i in xrange(signal.shape[1]):
        rval[:, i] = N.convolve(signal[:, i], kernel, 'same')
    return rval / kernel.sum()


def smooth(signal, window=5, kernel='gauss'):
    """smooth signal with kernel of type kernel and window size window

    :Parameters:
        signal : ndarray
            multichanneled signal [data, channel]
        window : ndarray
            window size of the smoothing filter (len(window) < signal
            .shape[0])
        kernel : ndarray
            kernel to use, one of
                - 'gauss': least squares
                - 'box': moving average
    """

    if kernel not in ['gauss', 'box']:
        raise ValueError('kernel must be in %s' % ['gauss', 'box'])
    filter_kernel = None
    if kernel == 'gauss':
        if window not in FILTER_KERNELS:
            raise ValueError('window must be in %s' % FILTER_KERNELS.keys())
        filter_kernel = FILTER_KERNELS[window]
    elif kernel == 'box':
        filter_kernel = N.ones(window)
    return _basic_smooth(signal, filter_kernel)

##--- MAIN

if __name__ == '__main__':
    pass
