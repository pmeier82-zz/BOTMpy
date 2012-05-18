"""builds filters and uses the prob model for the Chi^2 test"""

##---IMPORTS

from gen_spikes import DB, get_spikes
from spikepy.common import TimeSeriesCovE, mcvec_from_conc
from spikepy.nodes import (
    FilterBankNode, HomoscedasticClusteringNode, PCANode, PrewhiteningNode2)
from spikeplot  import plt, waveforms, cluster_projection
import scipy as sp

##---CONSTANTS

##---FUNCTIONS

def build_model(spks, ce, pca_dim=10):
    """cluster spikes and find a filter bank for the clustering"""

    print 'building model'

    # inits
    spks = sp.asarray(spks)
    tf, nc = ce.tf_max, ce.nc
    pca = PCANode(output_dim=pca_dim)
    whi = ce.get_whitening_op()
    clu = HomoscedasticClusteringNode(clus_type='gmm')

    # clustering
    spks_whi = sp.dot(spks, whi)
    spks_pca = pca(spks_whi)
    clu(spks_pca)

    # templates and filterbank
    lbls = clu.labels.astype(int)
    nclu = lbls.max() + 1
    wf, tp = {}, {}
    temps = sp.zeros((nclu, tf, nc))
    fb = FilterBankNode(tf=tf, ce=ce)
    for c in xrange(nclu):
        wf[c] = spks[lbls == c]
        tp[c] = wf[c].mean(0)
        fb.create_filter(mcvec_from_conc(tp[c], nc=nc), check=False)
    fb._check_internals()

    # return
    print 'finished building model - %s components' % fb.nfilter
    return fb, wf, tp


def component_divergence(obs, comps, ce, with_noise=False):
    """measure of divergence: mahalanobis distance"""

    print 'calculating divergence'

    # inits
    data = sp.asarray(obs, dtype=sp.float64)
    no, ns = data.shape
    if with_noise:
        comps = sp.vstack((comps, sp.zeros(data.shape[1])))
    comps = comps.astype(sp.float64)
    nc = comps.shape[0]
    isig = ce.get_icmx().astype(sp.float64)
    rval = sp.zeros((no, nc), dtype=sp.float64)

    # component wise divergence
    for n in xrange(no):
        x = data[n] - comps
        for c in xrange(comps.shape[0]):
            rval[n, c] = sp.dot(x[c], sp.dot(isig, x[c]))

    print 'finished calculating divergence'
    return rval


def prob_model(fb, spks, sig=0.05):
    """eyeballing the divergence model"""

    spks = sp.asarray(spks)
    comps = fb.get_template_set(mc=False)
    div = component_divergence(spks, comps, fb.ce)
    df = fb.tf * fb.nc
    div_max = int(div.max() * 1.05)
    chi2_proto = sp.stats.chi2.pdf(sp.arange(div_max), df)
    pmass = sp.stats.chi2.cdf(div, df)
    for i in xrange(fb.nfilter):
        # spike sets
        explain_i = pmass[:, i] <= 1.0 - sig
        spks_i_e = spks[explain_i == True]
        spks_i_ne = spks[explain_i == False]

        # figure
        f = plt.figure()
        f.suptitle('Unit %s - %s events' % (i, sum(explain_i == True)))

        # divergence histogram all
        ax = f.add_subplot(221)
        ax.hist(div[:, i], bins=50, normed=True)
        ax.plot(chi2_proto)

        # divergence histogram this unit
        ax = f.add_subplot(222)
        if len(div[explain_i == True, i]) > 0:
            ax.hist(div[explain_i == True, i], bins=50, normed=True)
        ax.plot(chi2_proto)

        # spike plots
        ax = f.add_subplot(223)
        ax.plot(spks_i_e.T, color='grey')
        ax.plot(comps[i], color='red')
        ax = f.add_subplot(224)
        ax.plot(spks_i_ne.T, color='grey')
        ax.plot(comps[i], color='red')

    # plot not explained events
    explain = (pmass <= 1.0 - sig).sum(1) == 0
    print len(explain), len(spks)
    spks_ne = spks[explain == False]

    # plot not explained spikes
    if spks_i_ne.size > 0:
        f = plt.figure()
        ax = f.add_subplot(211)
        ax.plot(spks_ne.T, color='grey')
        ax.plot(comps.T)
    else:
        print 'no spks_ne'

    plt.show()

##---MAIN

if __name__ == '__main__':
    # plotting
    plt.interactive(False)

    # intis
    tf = 47
    kv = [1, 3, 5, 7, 9]
    #kv = [6, 9, 13, 18]
    s_train, s_test, ce = get_spikes(5000, 1000, tf=tf,
                                     exp='L011', tet=3,
                                     det_kwargs={'kvalues':kv,
                                                 'tf':tf,
                                                 'min_dist':int(tf / 2)})

    # filterbank
    fb, wf, tp = build_model(s_train, ce)

    # waveforms and templates
    #waveforms(wf, templates=tp, plot_mean=True, show=False)

    # cluster projections
    whi = ce.get_whitening_op()
    wf_whi = {}
    for c in wf.keys():
        wf_whi[c] = sp.dot(wf[c], whi)
    cluster_projection(wf_whi, show=False)

    # prob model eval
    prob_model(fb, s_test)

    # end
    plt.show()
