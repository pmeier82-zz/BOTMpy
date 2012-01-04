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


"""type conversion node - used to do AD conversions"""

__docformat__ = 'restructuredtext'
__all__ = ['TypeConversionNode']

##--- IMPORTS

import scipy as sp
from .base_nodes import ResetNode

##--- CLASSES

class TypeConversionNode(ResetNode):
    """type conversion node"""

    ## constructor

    def __init__(self, val_range=100.0, output_dtype=sp.float32,
                 signed=False):
        """
        :Parameters:
            val_range : float
                the range of values of the projection space
            output_dtype : scipy.dtype
                dtype
            signed : bool
                is the input signed?
        """
        # super call
        super(TypeConversionNode, self).__init__()

        # init
        self.output_dtype = output_dtype
        self.val_range = val_range
        self.signed = bool(signed)

    ## node implementation

    def _get_supported_dtypes(self):
        if self.signed is True:
            return [sp.dtype(c) for c in sp.typecodes['Integer']]
        elif self.signed is False:
            return [sp.dtype(c) for c in sp.typecodes['UnsignedInteger']]
        else:
            raise ValueError('signed is not boolean')

    def is_trainable(self):
        return False

    def is_invertible(self):
        return False

    def _execute(self, x):
        # inits
        int_range = sp.iinfo(x.dtype).max
        mod = 2.0 * self.val_range / int_range
        print int_range

        # project to range
        rval = x.astype(self.output_dtype)
        if self.signed is False:
            rval -= int_range / 2.0
        rval *= mod

        # we need to reset the dtype!
        self._dtype = None

        # return
        return rval

##--- MAIN

if __name__ == '__main__':
    pass
