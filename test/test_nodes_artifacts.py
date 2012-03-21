# tests artifact detector

##---IMPORTS

try:
    import unittest2 as ut
except ImportError:
    import unittest as ut

from numpy.testing import assert_equal, assert_almost_equal
import scipy as sp
from spikepy.nodes import ArtifactDetectorNode

##---TESTS

class TestArtifactDetectorNode(ut.TestCase):
    def setUp(self):
        pass

    def testArtifactDetector(self):
        pass

    """
    from os import listdir, path as osp
    from spikeplot import mcdata, plt
    from spikepy.common import XpdFile
    from spikepy.nodes import SDMteoNode as SDET

    tf = 65
    AD = ArtifactDetectorNode()
    SD = SDET(tf=tf, min_dist=int(tf * 0.5))
    XPDPATH = '/home/phil/Data/Munk/Louis/L011'

    for fname in sorted(filter(lambda x:x.startswith('L011') and
                                        x.endswith('.xpd'),
                               listdir(XPDPATH)))[:20]:
        arc = XpdFile(osp.join(XPDPATH, fname))
        data = arc.get_data(item=7)
        AD(data)
        print AD.events
        print AD.get_nonartefact_epochs()
        print AD.get_fragmentation()
        SD(data)
        f = mcdata(data=data, other=SD.energy, events={0:SD.events},
                   epochs=AD.events, show=False)
        for t in SD.threshold:
            f.axes[-1].axhline(t)
        plt.show()
    """

if __name__ == '__main__':
    ut.main()
