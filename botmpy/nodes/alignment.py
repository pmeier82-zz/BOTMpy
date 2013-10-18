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

"""initialization - alignment of spike waveform sets"""
__docformat__ = 'restructuredtext'
__all__ = ['AlignmentNode']

## IMPORTS

import scipy as sp
from scipy.signal import resample
from .base_nodes import ResetNode

## CLASSES

class AlignmentNode(ResetNode):
    """aligns a set of spikes on the mean waveform of the set"""

    ## constructor

    def __init__(self, nchan=4, max_rep=32, max_tau=10, resample_factor=None,
                 cut_down=True, dtype=sp.float32, debug=False):
        """
        :Parameters:
            nchan : int
                channel count
                Default=4
            max_rep : int
                maximum repetitions
                Default=32
            max_tau : int
                upper bound for the shifting. will shift from -tau to +tau
                Default=10
            resample_factor : float or None
                before shifting, resample with this factor. after return
                resample with inverse of this factor, if None ignore
                Default=None
            cut_down: bool
                If True, cut down to original size, stripping the padding
                dimensions. If False, return with the padding dimensions.
                Default=True
            dtype : scipy.dtype
                dtype for the internal calculations
                Default=scipy.float32
            debug : bool
                If True, be verbose.
                Defult=False
        """

        # super
        super(AlignmentNode, self).__init__(dtype=dtype)

        # members
        self.nchan = int(nchan)
        self.tau = None
        self.spikes = None
        self.debug = bool(debug)
        self.max_rep = int(max_rep)
        self.max_tau = int(max_tau)
        self.resample_factor = None
        if resample_factor is not None:
            self.resample_factor = float(resample_factor)
        self.cut_down = bool(cut_down)

    ## node implementation

    def is_invertable(self):
        return False

    def is_trainable(self):
        return False

    def _reset(self):
        self.tau = None
        self.spikes = None

    def _execute(self, x):
        # inits
        n, dim = x.shape
        if n < 2:
            raise ValueError('too few spikes to align')

        self.spikes = sp.zeros((n, dim + 2 * self.max_tau * self.nchan))
        self.tau = sp.zeros(n)

        # put spikes in, resample and extrapolate
        idx_base = sp.arange(dim / self.nchan)
        spike_idx = []
        for c in xrange(self.nchan):
            spike_idx += (
                idx_base +
                c * dim / self.nchan +
                (2 * c + 1) * self.max_tau
                ).tolist()
        self.spikes[:, spike_idx] = x
        if self.resample_factor is not None:
            if self.debug is True:
                print 'upsampling by %f' % self.resample_factor
            self.spikes = resample(
                self.spikes,
                self.spikes.shape[1] * self.resample_factor,
                axis=1
            )
            self.max_tau *= self.resample_factor
            self.max_tau = int(self.max_tau)
            if self.debug is True:
                print 'upsampled size: %d, maxtau: %d' % (self.spikes
                                                          .shape[1],
                                                          self.max_tau)

        # get the mean spike and start iteration
        mean_spike = self.spikes.mean(axis=0)

        changes = sp.inf
        cur_rep = 0
        while cur_rep < self.max_rep and changes > n * 0.005:
            changes = 0
            q_avg = 0.0
            for s in xrange(n):
                # take the current spike out of the mean
                mean_spike -= self.spikes[s, :] / n
                # fit quality
                q_max = 0.0
                best_tau = 0
                for tau in xrange(-self.max_tau, self.max_tau + 1):
                    # shift the current spike and compute distance to the mean
                    shifted_spike = shift_row(self.spikes[s, :], tau)
                    q_tau = sp.absolute(sp.dot(mean_spike, shifted_spike))
                    if q_tau > q_max or q_max == 0.0:
                        best_tau = tau
                        q_max = q_tau
                if best_tau != 0:
                    # apply shift
                    self.spikes[s, :] = shift_row(self.spikes[s, :], best_tau)
                    self.tau[s] += best_tau
                    changes += 1
                q_avg += q_max

                # put the shifted spike back into the mean
                mean_spike += self.spikes[s, :] / n

            cur_rep += 1
            if self.debug is True:
                print '\t[%s] -> qual=%.4f (*%d)' % (
                cur_rep, q_avg / n, changes)

        # get rid of the padding and resampling
        if self.resample_factor is not None:
            if self.debug is True:
                print 'downsampling again'
            self.spikes = resample(
                self.spikes,
                self.spikes.shape[1] * 1.0 / self.resample_factor,
                axis=1
            )
            # correct taus
            self.tau = (self.tau / self.resample_factor).round().astype(int)
        if self.cut_down is True:
            self.spikes = self.spikes[:, spike_idx]

        # return aligned spikes
        return self.spikes

## HELPERS

def shift_row(row, shift):
    if shift == 0:
        return row
    if shift > 0:
        return sp.concatenate(([0] * shift, row[:-shift]))
    else:
        return sp.concatenate((row[-shift:], [0] * -shift))

## MAIN

if __name__ == '__main__':
    pass
