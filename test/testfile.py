# generic testing

from numpy.testing import assert_equal
import scipy as sp
import matplotlib
from spikepy.common import xcorr, overlaps

matplotlib.use('GtkAgg')
from matplotlib import pyplot

class DictDiffer(object):
    """
    Calculate the difference between two dictionaries as:
    (1) items added
    (2) items removed
    (3) keys same in both but changed values
    (4) keys same in both and unchanged values
    """

    def __init__(self, current_dict, past_dict):
        self.current_dict, self.past_dict = current_dict, past_dict
        self.set_current, self.set_past = set(current_dict.keys()), set(
            past_dict.keys())
        self.intersect = self.set_current.intersection(self.set_past)

    def added(self):
        return self.set_current - self.intersect

    def removed(self):
        return self.set_past - self.intersect

    def changed(self):
        return set(o for o in self.intersect if self.past_dict[o] !=
                                                self.current_dict[o])

    def unchanged(self):
        return set(o for o in self.intersect if self.past_dict[o] ==
                                                self.current_dict[o])


def main1():
    n = 100
    two_pi_ls = sp.linspace(0.0, 2 * sp.pi, 100)
    a = sp.sin(two_pi_ls)
    #    x = xc(a, b)
    xc_spikepy = xcorr(a)
    xc_pylab = pyplot.xcorr(a, a, maxlags=n - 1, normed=False)[1]
    xc_xrange = xrange(-n + 1, n)

    pyplot.ion()

    f = pyplot.figure()
    ax = f.add_subplot(411)
    ax.plot(xrange(n), a)
    ax = f.add_subplot(412)
    ax.plot(xc_xrange, xc_spikepy)
    ax = f.add_subplot(413)
    ax.plot(xc_xrange, xc_pylab)
    ax = f.add_subplot(413)
    xc_plt = ax.xcorr(xc_spikepy, xc_pylab, maxlags=n - 1)
    f.show()

    print xc_plt
    assert_equal(xc_spikepy, xc_pylab)


def main2():
    sts = {
        'A':sp.array([50, 150, 250]),
        'B':sp.array([51, 251, 300]),
        'C':sp.array([20, 200, 299])}
    sts_test = {
        'A':sp.array([True, False, True]),
        'B':sp.array([True, True, True]),
        'C':sp.array([False, False, True])}
    ovlp, ovlp_nums = overlaps(sts, 10)

    DD = DictDiffer(sts_test, ovlp)
    print DD.added()
    print DD.changed()
    print DD.unchanged()
    print DD.removed()

if __name__ == '__main__':
    # main1()
    main2()
