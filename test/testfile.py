# generic testing

from numpy.testing import assert_equal
import scipy as sp
import matplotlib
from spikepy.common import xcorr

matplotlib.use('GtkAgg')
from matplotlib import pyplot

if __name__ == '__main__':
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


