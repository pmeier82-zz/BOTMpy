"""generate spikes from a tetrode of the munk experiments in the database"""

##---IMPORTS

from spikedb import MunkSession
from spikepy.common import TimeSeriesCovE, MxRingBuffer
from spikepy.nodes import SDMteoNode

##---CONSTANTS

DB = MunkSession()
DB.connect()

##---FUNCTIONS

def get_spikes(n_train, n_test, exp='L014', tet=7, tf=47, det_kwargs={}):
    """

    :type n_train: int
    :param n_train: spikes to build the filters from
    :type n_test: int
    :param n_test: spikes to test against the filters
    :type exp: str
    :param exp: experiment as string, Default:L014
    :type tet: int
    :param tet: tetrode nr, Default:7
    :type det_kwargs: dict
    :param det_kwargs: keyword dictionary for the detector
    :return: tuple
    :rtype: train_spks, test_spks, ce
    """

    print 'getting spikes',

    # intis
    exp_id = DB.get_exp_id(exp)
    tet_id = DB.get_tetrode_id(exp_id, tet)
    tid_list = DB.get_trial_range_exp(exp_id)
    tid_list_idx = 0
    align_at = int(.25 * tf)
    ce = TimeSeriesCovE(tf_max=tf, nc=4)
    spks_train = MxRingBuffer(capacity=n_train, dimension=(tf * 4,))
    spks_test = MxRingBuffer(capacity=n_test, dimension=(tf * 4,))
    det_kwargs.update(tf=tf, threshold_factor=3.0)
    if 'kvalues' not in det_kwargs:
        det_kwargs['kvalues'] = [1, 3, 5, 7, 9]
    SD = SDMteoNode(**det_kwargs)

    print 'from', exp, ', tetrode', tet, '@tf=%d' % tf

    # db stuff
    while not spks_train.is_full:
        tid = tid_list[tid_list_idx]
        tid_list_idx += 1

        data = DB.get_tetrode_data(tid, tet_id)

        SD.reset()
        SD(data)

        spks = SD.get_extracted_events(mc=False, align_kind='min',
                                       align_at=align_at)
        spks_train.extend(spks)

        nep = SD.get_epochs(invert=True, merge=True)
        ce.update(data, epochs=nep)
    ntrials_train = tid_list_idx + 1
    print 'got train spikes from', ntrials_train, 'trials'

    while not spks_test.is_full:
        tid = tid_list[tid_list_idx]
        tid_list_idx += 1

        data = DB.get_tetrode_data(tid, tet_id)

        SD.reset()
        SD(data)

        spks = SD.get_extracted_events(mc=False, align_kind='min',
                                       align_at=align_at)
        spks_test.extend(spks)

        nep = SD.get_epochs(invert=True, merge=True)
        ce.update(data, epochs=nep)
    ntrials_test = tid_list_idx + 1 - ntrials_train
    print 'got test spikes from', ntrials_test, 'trials'

    # return
    return spks_train, spks_test, ce

##---MAIN

if __name__ == '__main__':
    from spikeplot  import waveforms, plt

    plt.interactive(False)

    tf = 47
    kv = [1, 3, 5, 7, 9]
    s_train, s_test, ce = get_spikes(500, 1000, tf=tf,
                                     det_kwargs={'kvalues':kv, 'tf':tf})

    waveforms(s_train[:], show=False)
    waveforms(s_test[:], show=False)
    plt.matshow(ce.get_cmx())
    plt.colorbar()

    plt.show()
