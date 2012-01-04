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


"""spiketrains alignment"""

__docformat__ = 'restructuredtext'
__all__ = ['align_spike_trains', 'similarity', 'simi', 'overlaps',
           'nice_table_from_analysis', 'csv_from_analysis', 'print_nice_table']

##--- IMPORTS

import scipy as sp
from common import dict_list_to_ndarray, dict_sort_ndarrays, matrix_argmax

##--- FUNCTIONS

# --------------------------------------------------------------------
def align_spike_trains(G, E, max_shift=15, max_jitter=6, max_overlap_dist=45):
    """computes the evaluation of spike sorting
    E contains the sorted spike
    trains - given the real/ideal/ground truth spike trains in G

    Calculates the similarity matrix between all pairs of spike trains from the
    ground truth and the estimation. This is used to find the optimal assignment
    between the spike trains, if one is a ground truth and the other is an
    estimation.

    Assignment Matrix:
        A label is assigned to every estimated spike. The following table lists
        all possible labels, given the different configurations is the
        ground truth. We assume E1 was found TO correlate with G1 and E2 is
        corresponding to G2. A "1" indicates a spike w.r.t. shift and jitter.

        =====  === === ===== == == ===== ====== ===== =====
        G1      1        1       1   1                  1
        G2          1    1                 1
        G3                           1     1      1
        =====  === === ===== == == ===== ====== ===== =====
        E1      1   1    1    1            1      1     1
        E2               1                        1     1
        =====  === === ===== == == ===== ====== ===== =====
        label  TP  FPA TPOvp FP FN FNOvp FPAOvp FPOGT

        TP : true positive
            E1 spike is assigned to associated ground truth spike train.
        FPA
            E1 spike is assigned to non associated ground truth spike train.
        TPOvp : true positive and overlap
            E1 spike is assigned to associated ground truth spike train and this
            spike participates in an overlap with another ground truth spike train.
        FP : false positive
            E1 spike is not assigned to any ground truth spike.
        FN : false negative
            There is no E1 spike for a spike in the associated ground truth
            spike train.
        FNOvp : false negative and overlap
            There is no E1 spike for a spike in the associated ground truth
            spike train and this spike participates in an overlap with another
            ground truth spike train.
        FPAOvp
            E1 spike is assigned to a spike of a non associated
            ground truth spike train and this spike participates in an
            overlap.

    :Returns:
        dict : dict with lots of entries [specify this!]

    :Parameters:
        G : dict of ndarray
            dict containing 1d ndarrays/lists of integers, representing the single unit
            spike trains. This is the ground truth.
        E : dict of ndarray
            dict containing 1d ndarrays/lists of integers, representing the single unit
            spike trains. this is the estimation.
        maxshift : int
            Upper bound for the tested shift of spike trains towards each other
            Default=15
        maxjitter : int
            upper bound for the tested jitter tolerance
            Default=6
        maxoverlapdistance : int
            upper bound for the tested overlap distance
            Default=45
    """
    # DOC: this documentation needs to be more precise on the return part!!

    # convert to ndarrays
    G = dict_list_to_ndarray(G)
    E = dict_list_to_ndarray(E)
    # sort the damned arrays!
    G = dict_sort_ndarrays(G)
    E = dict_sort_ndarrays(E)

    n = len(G)
    m = len(E)
    print n,m

    similarity_matrix = sp.zeros((n, m))
    shift_matrix = sp.zeros((n, m))

    print similarity_matrix

    rval = {'sfuncs':sp.zeros((n, m, 2 * max_shift + 1))}

    # Compute similarity score and optimal shift between all pairs of spiketrains
    for i in xrange(n):
        for j in xrange(m):
            sfunc = similarity(G[G.keys()[i]], E[E.keys()[j]], max_shift)
            similarity_matrix[i, j] = sfunc.max()
            shift_matrix[i, j] = sfunc.argmax() - max_shift
            rval['sfuncs'][i, j, :] = sfunc

    # Shift all estimated spike trains so that they fit optimal to the best
    # matching true spike train
    u_f2k = sp.zeros(m)
    delta_shift = sp.zeros(m)
    for j in xrange(m):
        myidx = similarity_matrix[:, j].argmax()
        delta_shift[j] = shift_matrix[myidx, j]
        E[E.keys()[j]] = sp.array(E[E.keys()[j]]) + delta_shift[j]

    # sort the spiketrain pairings according to their similarity measure
    # this ensures that the best matching spiketrains will get all
    # the matching spikes. no spike that matches will thus be aligned to
    # another spike train.
    sorted_tupels = []
    S = similarity_matrix.copy()
    for i in xrange(n * m):
        maxidx = S.argmax()
        sorted_tupels.append((int(sp.floor(maxidx / m)), maxidx % m))
        S[sorted_tupels[i]] = -1

    # init alignment dictonary
    alignment = {}
    idx = 0
    for i in xrange(n):
        for j in xrange(m):
            alignment[(G.keys()[i], E.keys()[j])] = []
            idx += 1

    # convert G and E to lists, otherwise we cant remove objects
    GBlocked = {};
    EBlocked = {}
    rval['num_known'] = sp.zeros(n)
    for i in xrange(n):
        rval['num_known'][i] = G[G.keys()[i]].shape[0]
        GBlocked[G.keys()[i]] = sp.zeros(G[G.keys()[i]].shape)
    rval['num_found'] = sp.zeros(m)
    for j in xrange(m):
        rval['num_found'][j] = E[E.keys()[j]].shape[0]
        EBlocked[E.keys()[j]] = sp.zeros(E[E.keys()[j]].shape)

    # GBlocked will contain for every _inserted_ spike a 0 or a 1
    # 0: it was not assigned to any of the found spike trains => false negatives
    # 1: it was assigned to a spike of E => either true positive or false
    #    negative + false positive wrong assignment
    #
    # EBlocked will contain for every _found_ spike a 0 or a 1
    # 0: it was not assigned to any of the inserted spike trains => false positive
    # 1: it was assigned to a spike of G => it will be handled when G is analyzed!

    spike_number_assignment_matrix = sp.zeros((n, m))
    # run over the sorted tupels and block all established spike assignments
    for i in xrange(n * m):
        k1idx = sorted_tupels[i][0]
        k2idx = sorted_tupels[i][1]
        k1 = G.keys()[sorted_tupels[i][0]]
        k2 = E.keys()[sorted_tupels[i][1]]
        train1 = G[k1]
        train2 = E[k2]
        idx1 = 0
        idx2 = 0
        while idx1 < len(train1) and\
              idx2 < len(train2):
            # if a spike is blocked it cannot be associated anymore. jump
            if GBlocked[k1][idx1] == 1:
                idx1 += 1
                continue
            if EBlocked[k2][idx2] == 1:
                idx2 += 1
                continue

            if train1[idx1] <= train2[idx2] + max_jitter and\
               train1[idx1] >= train2[idx2] - max_jitter:
                # spike assignment found, remove spikes
                alignment[(k1, k2)].append((idx1, idx2))
                GBlocked[k1][idx1] = 1
                EBlocked[k2][idx2] = 1

                spike_number_assignment_matrix[k1idx, k2idx] += 1
                # We cannot calculate TP/FP/FNs here, since we dont yet know
                # which G belongs to which E (see next step)
            if train1[idx1] < train2[idx2]:
                idx1 += 1
            else:
                idx2 += 1

    # now establish the one to one relationships between the true and
    # found spike trains. this is a different relationship than the one before
    # because there can maximal be min(n,m) associations. If there are more
    # found spike trains than inserted (m>n), some wont have a partner and
    # be thus treated as FPs. If there are more inserted than found (m>n) than
    # some will be treated as being not found (FNs).

    # Assignment vectors between true and estimated units
    u_k2f = sp.ones(n, dtype=sp.int16) * -1
    u_f2k = sp.ones(m, dtype=sp.int16) * -1

    nAssociations = min(n, m)
    found = 0
    count = 0
    blocked_rows = []
    blocked_cols = []
    snam = spike_number_assignment_matrix.copy()
    while (found < nAssociations) and (count < n * m):
        i, j = matrix_argmax(snam)
        if i not in blocked_rows and j not in blocked_cols:
            blocked_rows.append(i)
            u_k2f[i] = j
            blocked_cols.append(j)
            u_f2k[j] = i
            found += 1
        snam[i, j] = -1
        count += 1

    # now we want to calculate FPs and FNs. Since in spike_number_assignment_matrix
    # the assigned spikes are coded and in u_k2f and u_f2k the assignments of the
    # units to each other, we can now compare the number of assignments to the total
    # number of spikes in the corresponding trains. this will directly give the
    # correct/error numbers.

    # mark all the overlapping spikes
    ret = overlaps(G, max_overlap_dist)
    O = ret['O']
    NO = ret['Onums']

    # initialize dictionaries for labels for all spikes
    GL = {}
    for k in G.keys():
        GL[k] = sp.zeros(G[k].shape, dtype=sp.int16)
    EL = {}
    for k in E.keys():
        EL[k] = sp.zeros(E[k].shape, dtype=sp.int16)


    # run over every single spike and check which label it gets
    #             1      2      3     4       5     6      7
    labelList = ['TP', 'TPO', 'FP', 'FPA', 'FPAO', 'FN', 'FNO']
    TP = sp.zeros(n)
    TPO = sp.zeros(n)
    FP = sp.zeros(m) # m !!
    FPA = sp.zeros(n)
    FPAO = sp.zeros(n)
    FPA_E = sp.zeros(m) # m!!
    FPAO_E = sp.zeros(m) # m!!
    FN = sp.zeros(n)
    FNO = sp.zeros(n)
    # handle the spikes which were aligned first
    for i in xrange(n):
        k1 = G.keys()[i]
        for j in xrange(m):
            k2 = E.keys()[j]
            for a in xrange(len(alignment[(k1, k2)])):
                ovp = O[k1][alignment[(k1, k2)][a][0]]
                # labelList = ['TP', 'TPO', 'FP', 'FPA', 'FPAO', 'FN', 'FNO']
                if u_k2f[i] == j:
                    if ovp == 1:
                        GL[k1][alignment[(k1, k2)][a][0]] = 2 # TPO
                        EL[k2][alignment[(k1, k2)][a][1]] = 2
                        TPO[i] += 1
                    else:
                        GL[k1][alignment[(k1, k2)][a][0]] = 1 # TP
                        EL[k2][alignment[(k1, k2)][a][1]] = 1
                        TP[i] += 1
                else:
                    if ovp == 1:
                        GL[k1][alignment[(k1, k2)][a][0]] = 5 #FPAO
                        EL[k2][alignment[(k1, k2)][a][1]] = 5
                        FPAO[i] += 1
                        FPAO_E[j] += 1  # Count assignment errors twice! FP + FN
                    else:
                        GL[k1][alignment[(k1, k2)][a][0]] = 4 # FPA
                        EL[k2][alignment[(k1, k2)][a][1]] = 4
                        FPA[i] += 1
                        FPA_E[j] += 1 # Count assignment errors twice!

        # Now check all spikes of i which have no labels. those are FNs
        for spk in xrange(len(GL[k1])):
            if GL[k1][spk] == 0:
                if O[k1][spk] == 1:
                    GL[k1][spk] = 7  # FNO
                    FNO[i] += 1
                else:
                    GL[k1][spk] = 6  # FN
                    FN[i] += 1
        # The last thing to do is to check all labels of spikes in E. Those
    # which have no label yet are FPs
    for j in xrange(m):
        k2 = E.keys()[j]
        FP[j] = 0
        for spk in xrange(len(EL[k2])):
            if EL[k2][spk] == 0:
                EL[k2][spk] = 3 # FP
                FP[j] += 1


    # Build return value dictionary
    rval['scores'] = similarity_matrix
    rval['shifts'] = shift_matrix
    rval['delta_shift'] = delta_shift
    rval['alignment'] = alignment
    rval['overlapping'] = O
    rval['spike_number_assignment_matrix'] = spike_number_assignment_matrix
    rval['EL'] = EL
    rval['GL'] = GL

    rval['TP'] = TP
    rval['TPO'] = TPO
    rval['FPA'] = FPA
    rval['FPAO'] = FPAO
    rval['FN'] = FN
    rval['FNO'] = FNO
    rval['FP'] = FP

    rval['u_k2f'] = u_k2f
    rval['u_f2k'] = u_f2k

    # build pretty table
    rval['table'] = []
    rval['table'].append(['GT Unit ID', # ID of known Unit
                          'Found Unit ID', # ID of associated found Unit
                          'Known Spikes', # Number of Spikes of Known Unit
                          'Overlapping Spikes', # Number of those Spikes participating in an Overlap
                          'Found Spikes', # Number of Spikes of Found Unit
                          'True Pos', # Number of those Spikes which are assigned to Spikes
                          # of the associated known Unit.
                          'True Pos Ovps', #
                          'False Pos Assign GT', # Number of Spikes of Found Unit which are assigned to
                          # to a non-associated Unit
                          'False Pos Assign Found', #
                          'False Pos Ovps GT', # FPAO
                          #
                          'FPs Assign Ovps Found', # FPAO_E
                          'False Neg', #
                          'False Neg Overlaps', #
                          'False Pos'])              #

    # Build Table with one row for every assignment of two spike trains and one row
    # for every unassigned spike train. Problem TODO: The number of false positive assignments for
    # one of the assignment rows has to be the sum of the individual assignement
    # errors. ???? ->This way an assignment error counts as 2 errors (one FP and one FN).

    remaining_found_units = sp.ones(m)
    # Build the assignment rows and unassigned ground truth spike train rows first
    for i in xrange(n):
        unitk = G.keys()[i]
        known = rval['num_known'][i]
        overlapping = NO[i]
        tp = TP[i]
        tpo = TPO[i]
        fn = FN[i]
        fno = FNO[i]
        fpa = FPA[i]
        fpao = FPAO[i]

        j = rval['u_k2f'][i]
        unitf = ''
        found = fp = fpae = fpaoe = 0
        if j >= 0:
            remaining_found_units[j] = 0
            unitf = E.keys()[j]
            found = rval['num_found'][j]
            fp = FP[j]
            fpae = FPA_E[j]
            fpaoe = FPAO_E[j]

        rval['table'].append([unitk, unitf, known, overlapping, found, tp,
                              tpo, fpa, fpae, fpao, fpaoe, fn, fno, fp])

    # Append "False Positive Unit" which has all found spikes of found units which
    # were not assigned to a ground truth unit
    for j in xrange(m):
        if remaining_found_units[j] == 1:
            unitk = ''
            known = overlapping = tp = tpo = fn = fno = fpa = fpao = 0
            unitf = E.keys()[j]
            found = rval['num_found'][j]
            fp = FP[j]
            fpae = FPA_E[j]
            fpaoe = FPAO_E[j]
            rval['table'].append([unitk, unitf, known, overlapping, found, tp,
                                  tpo, fpa, fpae, fpao, fpaoe, fn, fno, fp])

    return rval


def similarity(st1, st2, mtau):
    """
    Calculates the crosscorrelation function between two spike trains
    """
    sfunc = sp.zeros(2 * mtau + 1)
    for tau in xrange(-mtau, mtau + 1):
        sfunc[tau + mtau] = simi(st1, st2 + tau)
    return sfunc


def simi(s1, s2):
    """Calculates the normalized scalar product between two binary vectors
    which are given by two point processes without actually creating the
    binary vectors.
    """
    p = 0
    s1idx = 0
    s2idx = 0
    while s1idx < s1.shape[0] and s2idx < s2.shape[0]:
        if s1[s1idx] == s2[s2idx]:
            p = p + 1
        if s1[s1idx] < s2[s2idx]:
            s1idx += 1
        else:
            s2idx += 1
    return 2.0 * p / (s1.shape[0] + s2.shape[0])

def overlaps(G, window):
    """
    Calculates a "boolean" dictonary, indicating for every spike in every
    spiketrain in G whether it belongs to an overlap or not
    """
    n = len(G)
    O = {}
    for k in G.keys():
        O[k] = sp.zeros(G[k].shape, dtype=sp.bool_)
    Onums = sp.zeros(len(G))
    # run over all pairs of spike trains in G
    for i in xrange(n):
        for j in xrange(i + 1, n):
            # for every pair run over all spikes in i and check whether a spike
            # in j overlaps
            trainI = G[G.keys()[i]]
            trainJ = G[G.keys()[j]]
            idxI = 0
            idxJ = 0
            while idxI < len(trainI) and idxJ < len(trainJ):
                # Overlapping?
                if abs(trainI[idxI] - trainJ[idxJ]) < window:
                    # Every spike can only be in one or no overlap. prevents triple
                    # counting
                    if O[G.keys()[i]][idxI] == 0:
                        O[G.keys()[i]][idxI] = 1
                        Onums[i] += 1
                    if O[G.keys()[j]][idxJ] == 0:
                        O[G.keys()[j]][idxJ] = 1
                        Onums[j] += 1

                if trainI[idxI] < trainJ[idxJ]:
                    idxI += 1
                else:
                    idxJ += 1
    ret = {'O':O, 'Onums':Onums}
    return ret

# -------------------------------------------------------------------
# Merges 2 dictonaries of spike trains and updates the corresponding alignment
#def merge_spiketrains(G, E, alignment = None):
#    rval = {}
#    for i in xrange(G.keys.shape[0]):
#        rval[G.keys[i]+100] = G[G.keys[i]]
#    for i in xrange(E.keys.shape[0]):
#        rval[E.keys[i]+200] = E[E.keys[i]]
#
#    if alignment is not None:

def nice_table_from_analysis(ana):
    """yields a nicely readable string that contains the information about
    the performance of that sorting"""
    rval = "GT ID  - FU ID |    KS    OS    FS    TP   TPO   FPA  FPAE  FPAO FPAOE    FN   FNO    FP\n"
    format_str = "%6s -%6s | %5d %5d %5d %5d %5d %5d %5d %5d %5d %5d %5d %5d\n"
    for i in xrange(len(ana['table']) - 1):
        rval = ''.join([rval, format_str % tuple(ana['table'][i + 1][0:14])])
    return rval


def csv_from_analysis(ana, header=True):
    """yields a string that can be stored as a .csv file"""
    rval = ""
    if header:
        rval = "GT ID  , FU ID ,    KS,    OS,    FS,    TP,   TPO,   FPA,  FPAE,  FPAO, FPAOE,    FN,   FNO,    FP\n"

    format_str = "%6s,%6s,%5d, %5d, %5d, %5d, %5d, %5d, %5d, %5d, %5d, %5d, %5d, %5d\n"
    for i in xrange(len(ana['table']) - 1):
        rval = ''.join([rval, format_str % tuple(ana['table'][i + 1][0:14])])
    return rval


def print_nice_table(ret):
    print nice_table_from_analysis(ret)

##--- MAIN

if __name__ == '__main__':
    #===========================================================================
    # from pydevd import set_pm_excepthook
    # set_pm_excepthook()
    #===========================================================================
    G = {'0':sp.array([1, 100, 200, 301, 400]), 1:sp.array([50, 150, 250, 303, 350])}

    E = {0:sp.array([1, 100, 200, 305, 307]), 1:sp.array([50, 150, 250, 590, 550, 648, 720])}

    ret = align_spike_trains(G, E, max_shift=2, max_jitter=12)
    from plot import P, spike_trains

    fig = P.figure(facecolor='white')
    spike_trains(G, spiketrains2=E, alignment=ret['alignment'], label1=ret['GL'], label2=ret['EL'], plot_handle=fig,
                 samples_per_second=16000)
    print 'Done Plot 0.'
    print ret['alignment']

#===============================================================================
#    G = {}
#    G['0'] = sp.array([40, 80, 90, 170, 400])
#    G[1] = sp.array([42, 150, 180, 190, 350])
#    G[2] = sp.array([39, 80, 150, 405])
#
#    E = {}
#    E[0] = sp.array([42, 80, 170, 401])
#    E[1] = sp.array([40, 90, 150, 180, 190, 250, 348, 420])
#
#    #ret = align_spike_trains(G, E, maxshift=2, maxjitter=2)
#   # print ret
#    import common.plot as plot
#
#    #print 'lala 1'
#    #plot.spike_trains(G, spiketrains2=E, alignment=ret['alignment'], show=0)
#
#    print 'Done Plot 1.'
#
#    G = {}
#    G['0'] = sp.array([40, 80, 90, 170, 400])
#    G[1] = sp.array([42, 150, 180, 190, 350])
#    G[2] = sp.array([39, 80, 150, 405])
#
#    E = {}
#    E[3] = sp.array([42, 80, 170, 401])
#    E['Multi Unit 1'] = sp.array([40, 90, 150, 180, 190, 250, 348, 420])
#    E[4] = sp.array([37, 79, 96, 149, 201, 405])
#    E['Multi Unit 2'] = sp.array([10, 20, 30, 40, 50, 60, 60, 170])
#
#    ret = align_spike_trains(G, E, maxshift=2, maxjitter=2, maxoverlapdistance=5)
#    print ret
#
#    from plot import P, spiketrains
#    fig = P.figure(facecolor='white')
#    spike_trains(G, spiketrains2=E, alignment=ret['alignment'], label1=ret['GL'], label2=ret['EL'], plot_handle=fig, samples_per_second=16000)
#    print 'Done Plot 2.'
#===============================================================================
