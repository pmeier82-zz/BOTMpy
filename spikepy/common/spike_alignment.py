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


"""spikes alignment functions"""
__docformat__ = 'restructuredtext'
__all__ = ['get_tau_for_alignment', 'get_tau_align_min', 'get_tau_align_max',
           'get_tau_align_energy', 'get_aligned_spikes']

##--- IMPORTS

import scipy as sp
from .util import INDEX_DTYPE
from .funcs_general import mcvec_to_conc
from .funcs_spike import epochs_from_spiketrain, get_cut, extract_spikes

##---FUNCTIONS

def get_tau_for_alignment(spikes, align_at):
    """return the per spike offset in samples (taus) of the maximum values to
    the desired alignment sample within the spike waveform.

    :type spikes: ndarray
    :param spikes: stacked spike waveforms in their multichanneled
        representation [nspikes, tf, nc]
    :type align_at: int
    :param align_at: sample to align the maximum at
    :returns: ndarray - offset per spike
    """

    dchan = [spike.max(0).argmax() for spike in spikes]
    tau = [spikes[i, :, dchan[i]].argmax() - align_at
           for i in xrange(spikes.shape[0])]
    return - sp.asarray(tau, dtype=INDEX_DTYPE)


def get_tau_align_min(spikes, align_at):
    """align on minimum"""

    return get_tau_for_alignment(-spikes, align_at)


def get_tau_align_max(spikes, align_at):
    """align on maximum"""

    return get_tau_for_alignment(spikes, align_at)


def get_tau_align_energy(spikes, align_at):
    """align on peak in the spike energy"""

    return get_tau_for_alignment(spikes * spikes, align_at)


def get_aligned_spikes(data, spiketrain, cut, align_at=-1, mc=True,
                       kind='none'):
    """return the set of aligned spikes waveforms and thei taus

    :type data: ndarray
    :param data: data with channels in the columns
    :type spiketrain: ndarray or list
    :param spiketrain: spike train of events in data
    :type cut: tuple or int
    :param cut: (cut_left,cut_right) tuple or int for symmetric cut tuple
    :type align_at: int
    :param align_at: align chosen feature at this sample in the waveform
    :type mc: bool
    :param mc: if True, return multichanneled waveforms, else return
        concatenated waveforms.
        Default=True
    :type kind: str
    :param kind: String giving the type of alignment to conduct. One of:

            - "max"    - align on maximum of the waveform
            - "min"    - align on minimum of the waveform
            - "energy" - align on peak of energy
            - "none"   - no alignment

        Default='none'
    :rtype: ndarray, ndarray
    :returns: stacked spike events, spike train with events corrected for
        alignment
    """

    if len(data) == 0:
        return sp.zeros(sum(cut))

    ep, st = epochs_from_spiketrain(spiketrain,
                                    cut,
                                    end=data.shape[0],
                                    with_corrected_st=True)
    if ep.shape[0] > 0:
        if kind in ['min', 'max', 'energy']:
            spikes = extract_spikes(data, ep, mc=True)
            tau = {'min':get_tau_align_min,
                   'max':get_tau_align_max,
                   'energy':get_tau_align_energy}[kind](spikes, align_at)
            ep, st = epochs_from_spiketrain(st - tau,
                                            cut,
                                            end=data.shape[0],
                                            with_corrected_st=True)
            spikes = extract_spikes(data, ep, mc=mc)
        else:
            spikes = extract_spikes(data, ep, mc=mc)
    else:
        cut = get_cut(cut)
        if mc is True:
            size = 0, sum(cut), data.shape[1]
        else:
            size = 0, sum(cut) * data.shape[1]
        spikes = sp.zeros(size)
    return spikes, st

##--- MAIN

if __name__ == '__main__':
    # initial imports and constants
    from spikeplot import plt, waveforms

    TF = 65
    OFF = 20
    KIND = 'min'

    # get a spikes train
    from spikedb import MunkSession

    DB = MunkSession()
    data = DB.get_tetrode_data(1, 3)
    from spikepy.nodes import SDMteoNode

    SD = SDMteoNode(tf=TF, threshold_factor=3.5)
    SD(data)
    st = SD.events
    ep, st = epochs_from_spiketrain(st, get_cut(TF, OFF),
                                    with_corrected_st=True)

    # plot raw
    spikes_raw = []
    for i in xrange(st.size):
        spikes_raw.append(mcvec_to_conc(data[ep[i][0]:ep[i][1]]))
    spikes_raw = sp.vstack(spikes_raw)
    waveforms(spikes_raw, tf=TF, title='RAW SPIKES', show=False)

    # plot aligned
    spikes_aligned, st = get_aligned_spikes(data,
                                            SD.events,
                                            get_cut(TF, OFF),
                                            20,
                                            mc=False,
                                            kind=KIND)

    # show aligned spikes
    waveforms(spikes_aligned, tf=TF, title='ALIGNED SPIKES ["%s"]' % KIND,
              show=False)
    for c in xrange(4):
        plt.axvline(c * TF + 20)
    plt.show()
