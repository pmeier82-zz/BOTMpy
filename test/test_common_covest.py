# tests for covariance estimation

import scipy as sp
from spikepy.common import TimeSeriesCovE

if __name__ == '__main__':
    dlen = 10000
    tf_max = 67
    nc = 4

    my_data = [sp.randn(dlen, nc) * (sp.arange(4) + 1),
               sp.randn(dlen, nc) * (sp.arange(4) + 5),
               sp.randn(dlen, nc) * (sp.arange(4) + 9)]

    E = TimeSeriesCovE(tf_max=tf_max, nc=4)
    E.new_chan_set((0, 1, 2, 3))
    E.new_chan_set((1, 2))
    E.update(my_data[0])
    E.update(my_data[1])
    E.update(my_data[1], epochs=[[0, 100], [1000, 5000], [9500, 9745]])
    print E

    Calltf67_params = {'tf':67, 'chan_set':(0, 1, 2, 3)}
    Calltf67 = E.get_cmx(**Calltf67_params)
    print Calltf67
    print Calltf67.shape
    print E.get_svd(**Calltf67_params)
    print E.get_cond(**Calltf67_params)

    C12tf67_params = {'tf':20, 'chan_set':(1, 2)}
    C12tf67 = E.get_cmx(**C12tf67_params)
    print C12tf67
    print C12tf67.shape
    print E.get_svd(**C12tf67_params)
    print E.get_cond(**C12tf67_params)

    iC12tf67 = E.get_cmx(**C12tf67_params)
    print iC12tf67
    print iC12tf67.shape

    whiC12tf67 = E.get_whitening_op(**C12tf67_params)
    print whiC12tf67
    print whiC12tf67.shape

    from spikeplot import plt

    plt.matshow(Calltf67)
    plt.colorbar(ticks=range(16))
    plt.matshow(C12tf67)
    plt.colorbar(ticks=range(16))
    plt.matshow(iC12tf67)
    plt.colorbar(ticks=range(16))
    plt.matshow(whiC12tf67)
    plt.colorbar(ticks=range(16))
    plt.show()
