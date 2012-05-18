##---IMPORTS

from spikepy.common import TimeSeriesCovE, mcvec_from_conc, get_aligned_spikes
from spikepy.nodes import SDMteoNode, BOTMNode, ABOTMNode, PCANode
from spikepy.nodes import HomoscedasticClusteringNode
from spikedb import MunkSession
import scipy as sp
import cPickle
from numpy.linalg import pinv, slogdet


##---MAIN

if __name__ == '__main__':
    FILE_T = '/home/phil/Data/test/session.npy'
    FILE_C = '/home/phil/Data/test/session.pkl'

    templates = None
    ce = None
    db = MunkSession()
    db.connect()
    data = db.get_tetrode_data(13013, 119)

    try:
        templates = sp.load(FILE_T)
        ce = cPickle.load(open(FILE_C, 'r'))
    except Exception, ex:
        print ex
        SD = SDMteoNode(threshold_factor=3.0)
        SD(data)
        spikes = SD.get_extracted_events()
        nep = SD.get_epochs(merge=True, invert=True)
        ce = TimeSeriesCovE(tf_max=47, nc=4)
        ce.update(data, epochs=nep)
        spikes_prw = sp.dot(spikes, ce.get_whitening_op(tf=47))
        PCA = PCANode(output_dim=15)
        spikes_prw_pca = PCA(spikes_prw)
        CLS = HomoscedasticClusteringNode(clus_type='gmm', debug=True)
        CLS(spikes_prw_pca)
        ntemps = CLS.labels.max() + 1
        temps = sp.zeros((ntemps, 47 * 4))
        spd = {}
        for i in xrange(ntemps):
            spd[i] = spikes[CLS.labels == i]
            temps[i] = spikes[CLS.labels == i].mean(0)
        templates = sp.zeros((ntemps, 47, 4))
        for i in xrange(ntemps):
            templates[i] = mcvec_from_conc(temps[i], nc=4)
        sp.save(FILE_T, templates)
        cPickle.dump(ce, open(FILE_C, 'w'))

    FB = ABOTMNode(templates=templates, ce=ce)
    FB(data)

    pp = FB.posterior_prob(FB.template_set)
    print pp
    print pp.argmax(1)

    pval = 0.005
    spks = FB.det.get_extracted_events(mc=False, align_kind='min')
    spks_cdf = FB.component_chi2_cdf(spks)
    print spks_cdf
    print sp.any(spks_cdf < 1. - pval, axis=1)
