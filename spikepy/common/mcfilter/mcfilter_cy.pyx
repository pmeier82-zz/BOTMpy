##---IMPORTS

import cython
import numpy as np

cimport numpy as np

##---FUNCTIONS

@cython.boundscheck(False)
@cython.wraparound(False)
def _mcfilter_hist_cy32_forfrom(
np.ndarray[np.float32_t, ndim=2] mc_data,
np.ndarray[np.float32_t, ndim=2] mc_filt,
np.ndarray[np.float32_t, ndim=2] mc_hist):
    cdef:
        unsigned int nc = mc_data.shape[1]
        unsigned int td = mc_data.shape[0]
        unsigned int tf = mc_filt.shape[0]
        unsigned int th = mc_hist.shape[0]
        np.ndarray[np.float32_t, ndim=1] fout
        np.ndarray[np.float32_t, ndim=2] data
        np.float32_t value
        unsigned int t, tau, c
    data = np.vstack((mc_hist, mc_data))
    fout = np.empty(td, dtype=np.float32)
    with nogil:
        for t from 0 <= t < td:
            value = 0.0
            for c from 0 <= c < nc:
                for tau from 0 <= tau < tf:
                    value += data[t + tau, c] * mc_filt[tau, c]
            fout[t] = value
        for t from 0 <= t < th:
            for c from 0 <= c < nc:
                mc_hist[t, c] = data[td + t, c]
    return fout, mc_hist

@cython.boundscheck(False)
@cython.wraparound(False)
def _mcfilter_hist_cy32(
np.ndarray[np.float32_t, ndim=2] mc_data,
np.ndarray[np.float32_t, ndim=2] mc_filt,
np.ndarray[np.float32_t, ndim=2] mc_hist):
    cdef:
        unsigned int nc = mc_data.shape[1]
        unsigned int td = mc_data.shape[0]
        unsigned int tf = mc_filt.shape[0]
        unsigned int th = mc_hist.shape[0]
        np.ndarray[np.float32_t, ndim=1] fout
        np.ndarray[np.float32_t, ndim=2] data
        np.float32_t value
        unsigned int t, tau, c
    data = np.vstack((mc_hist, mc_data))
    fout = np.empty(td, dtype=np.float32)
    with nogil:
        for t in range(td):
            value = 0.0
            for c in range(nc):
                for tau in range(tf):
                    value += data[t + tau, c] * mc_filt[tau, c]
            fout[t] = value
        for t in range(th):
            for c in range(nc):
                mc_hist[t, c] = data[td + t, c]
    return fout, mc_hist

@cython.boundscheck(False)
@cython.wraparound(False)
def _mcfilter_hist_cy64(
np.ndarray[np.float64_t, ndim=2] mc_data,
np.ndarray[np.float64_t, ndim=2] mc_filt,
np.ndarray[np.float64_t, ndim=2] mc_hist):
    cdef:
        unsigned int nc = mc_data.shape[1]
        unsigned int td = mc_data.shape[0]
        unsigned int tf = mc_filt.shape[0]
        unsigned int th = mc_hist.shape[0]
        np.ndarray[np.float64_t, ndim=1] fout
        np.ndarray[np.float64_t, ndim=2] data
        np.float64_t value
        unsigned int t, tau, c
    data = np.vstack((mc_hist, mc_data))
    fout = np.empty(td, dtype=np.float64)
    with nogil:
        for t in range(td):
            value = 0.0
            for c in range(nc):
                for tau in range(tf):
                    value += data[t + tau, c] * mc_filt[tau, c]
            fout[t] = value
        for t in range(th):
            for c in range(nc):
                mc_hist[t, c] = data[td + t, c]
    return fout, mc_hist

if __name__ == '__main__':
    pass
