##---IMPORTS

from spikepy.common import (TimeSeriesCovE, mcvec_from_conc,
                            get_aligned_spikes, VERBOSE)
from spikepy.nodes import (SDMteoNode, BOTMNode, ABOTMNode, PCANode,
                           HomoscedasticClusteringNode)
from spikedb import MunkSession
from spikeplot import plt
import scipy as sp
import cPickle
from numpy.linalg import pinv, slogdet

plt.interactive(False)

##---MAIN

if __name__ == '__main__':
    EID = 1
    TID = 3

    db = MunkSession()
    db.connect()

    FILE_T = '/home/phil/Data/test/session.npy'
    FILE_C = '/home/phil/Data/test/session.pkl'
    FILE_T2 = '/home/phil/Data/test/session_data_flow_templates.npy'
    FILE_C2 = '/home/phil/Data/test/session_data_flow_covest.pkl'

    templates = None
    ce = None
    db = MunkSession()
    db.connect()

    TF = 47
    ce = TimeSeriesCovE(tf_max=TF)

    FB = ABOTMNode(tf=TF, ce=ce, det_limit=5000, verbose=1)
    SD = SDMteoNode(tf=TF, kvalues=[1, 3, 5, 7, 9], threshold_factor=3.0)

    # building filter bank
    print 'starting template finding'
    try:
        templates = sp.load(FILE_T2)
        ce = cPickle.load(open(FILE_C2, 'r'))
        FB = BOTMNode(templates=templates, ce=ce, verbose=1)
    except Exception, ex:
        for tid in db.get_trial_range_exp(EID):
            data = db.get_tetrode_data(tid, TID)
            FB(data)
            if FB.nfilter:
                sp.save(FILE_T2, FB.template_set)
                cPickle.dump(ce, open(FILE_C2, 'w'))
                print 'done building filter bank'
                break
    FB.plot_template_set(show=True)

    print 'starting detection'
    chi2 = sp.stats.chi2
    if not hasattr(FB, '_learn_template'):
        align_at = int(0.25 * FB.tf)
    else:
        align_at = FB._learn_templates
    for tid in db.get_trial_range_exp(EID, limit=20):
        print 'trial:', tid, db.get_fname_for_id(tid)
        data = db.get_tetrode_data(tid, TID)

        FB(data)
        FB.plot_sorting_waveforms(show=False)

        SD.reset()
        SD(data)

        spks = SD.get_extracted_events(
            mc=False, align_kind='min', align_at=align_at)
        spks_div = FB.component_divergence(spks)
        if sp.any(spks_div < sp.inf):
            print 'all inf, skiping'
            continue
        sdrange = sp.arange(int(spks_div.max() + 1))

        f = plt.figure()
        for i in xrange(FB.nfilter):
            a = f.add_subplot(2, FB.nfilter, 2 * i + 1)
            a.hist(spks_div[:, i], normed=True)
            a.plot(sdrange, chi2.pdf(sdrange, FB.tf * FB.nc))

            st = FB.rval[i]
            if len(st) > 0:
                continue
                spks_u = get_aligned_spikes(
                    data, st, TF, align_at, mc=False, kind='min')[0]
                spks_u_div = FB.component_divergence(spks_u)

                a = f.add_subplot(2, FB.nfilter, 2 * i + 2)
                a.hist(spks_u_div[:, i], normed=True)
                sudrange = sp.arange(int(spks_u_div.max() + 1))
                a.plot(sudrange,
                       chi2.pdf(sudrange, FB.tf * FB.nc))
            else:
                f.add_subplot(2, FB.nfilter, 2 * i + 2)
        plt.show()
