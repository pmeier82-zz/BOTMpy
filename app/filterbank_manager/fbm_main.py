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


"""python module managing an online filter bank"""

__docformat__ = 'restructuredtext'
__all__ = ['FilterBankManager']

##---IMPORTS

##---IMPORTS

import scipy as sp
from fbm_group import FilterGroup
from blockstream import (load_blockstream, BS3Reader, WAVEProtocolHandler,
                         COVEProtocolHandler, Queue, Empty, USE_PROCESS,
                         BS3SortSetupBlock)
from common import get_cut, mcvec_to_conc
import traceback

##---CONSTANTS

SIGNIFICANT = 0.05

##---CLASSES

class FilterBankManager(object):
    """manager class updating the template set, managing deletion,spawning and
    update of templates and filter calculation for channel group.
    """

    def __init__(self, **kwargs):
        """
        :keywords:
            FILTER_GROUP_PARAMETERS
            srate : float (32000.0)
                sample rate of input data in Hz
            tf : float or int (2.5)
                template/filter length
                if float will be interpreted in ms.
                if int will be interpreted in samples.
            nc : int
                number of channels
            train_amount : int (1000)
                amount of multi-unit observations to train the model from
            spike_amount : int (300)
                amount of single unit observations to update the templates
            unit_timeout_ms : float
                time in ms after that a unit will be considered for removal
            ali_at_frac : int (6)
                fraction of the template length from the start where the
                minimum template sample should be aligned
            sigma : float
                spherical relaxation of the cluster variance in noise std
        """

        # filter group parameters
        self.params = {}
        self.params['srate'] = kwargs.get('srate', 32000.0)
        tf = kwargs.get('tf', 2.5)
        if type(tf) not in [int, float]:
            tf = 2.5
        if isinstance(tf, float):
            self.params['tf'] = int(self.params['srate'] / 1000.0 * tf)
        if isinstance(tf, int):
            self.params['tf'] = tf
        self.params['cut'] = kwargs.get('cut', get_cut(self.params['tf']))
        self.params['nc'] = kwargs.get('nc', 4)
        self.params['cs'] = tuple(range(self.params['nc']))
        self.params['train_amount'] = kwargs.get('train_amount', 1000)
        self.params['spike_amount'] = kwargs.get('spike_amount', 300)
        self.params['spike_timeout'] = int(
            self.params['srate'] / 1000.0 *
            kwargs.get('unit_timeout_ms', 10000))
        ali_at_frac = float(kwargs.get('align_at_frac', 7))
        self.params['ali_at'] = int(sum(self.params['cut']) / ali_at_frac)
        self.params['sigma'] = kwargs.get('sigma', 4.0)

        # internals
        self.groups = {}

        # internals
        self.status = 0

    ## properties

    def add_filter_group(self, fg_idx=None):
        """sets up new filter group and returns True if successful"""

        if fg_idx is None:
            fg_idx = max(self.groups.keys()) + 1
        if fg_idx in self.groups:
            return False
        else:
            self.groups[fg_idx] = FilterGroup(fg_idx)
            return True

    ## methods

    def train(self, fg_idx, data):
        """train the model for that filter group

        :type fg_idx: int
        :param fg_idx: filter group index
        :type data: ndarray
        :param data: mc data
        :return bool: True if training has ended
        """

    def update(self, fg_idx, data, sort_sts, det_st=None):
        """update the manager with a strip of data and the
        corresponding
        sorted
        spike train

        :type fg_idx: int
        :param fg_idx: filter group idx
        :type data: ndarray
        :param data: input data strip
        :type sort_sts: dict of ndarray
        :param sort_sts: sorted spike train set
        :type det_st: ndaraay
        :param det_st: detected spike train, will be estimated if not
        given
        """

        # process inputs
        if det_st is None:
            det_st = self.det(data)
        spikes_det = get_aligned_spikes(data,
                                        det_st,
                                        self.params['cut'],
                                        self.params['align_at'],
                                        mc=True,
                                        kind=self.params['ali_kind'])
        spikes_sort = {}
        for k in sort_sts:
            spikes_sort[k] = get_aligned_spikes(data,
                                                sort_sts[k],
                                                self.params['cut'],
                                                self.params['align_at'],
                                                mc=True,
                                                kind=self.p
            arams[
            'ali_kind'])

            def matching(self, fg_idx):
                pass

        ##---MAIN

        if __name__ == '__main__':
            # build data
            means = [(20, 20), (30, 40), (0, 0)]
            X = sp.randn(1000, 2)
            X[:500] += means[0]
            X[500:] += means[1]
