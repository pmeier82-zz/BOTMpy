##---IMPORTS

import sys
from spikedb import MunkSession
import scipy as sp
from tables import openFile

##---CONSTANTS

db = MunkSession()
db.connect()

##---FUNCTIONS

def get_whitening_ops(C):
    """produce the whitening operators w.r.t. to svd and cholesky

    :type C: ndarray
    :param C: symetric pos. semi-def. matrix (covariance matrix)
    :rtype: tuple
    :returns: white_svd, white_chol
    """

    svdC = sp.linalg.svd(C)
    white_svd = sp.dot(sp.dot(svdC[0], sp.diag(sp.sqrt(1. / svdC[1]))),
                       svdC[2])
    condC = svdC[1].max() / svdC[1].min()
    cholC = sp.linalg.cholesky(C)
    white_chol = sp.linalg.inv(cholC)
    return condC, white_svd, white_chol


def get_cov(id_ana):
    q = db.query("""
    SELECT
      a.id, t.id
    FROM
      analysis a
      JOIN experiment e ON a.expid=e.id
      JOIN block b ON b.expid=e.id
      JOIN trial t ON t.block=b.id
    WHERE
      a.id=%s AND t.trialidx=a.trialidxstart AND a.kind='SORT'
    ORDER BY t.id
    LIMIT 1
    """ % id_ana)
    if len(q) == 0:
        return None
    ce = db.get_covariance_estimator(q[0][0], q[0][1])
    if ce is not None:
        return ce.get_cmx(tf=ce.tf_max)
    else:
        return None


def get_sample(C, n=100000):
    d = C.shape[0]
    return sp.random.multivariate_normal(sp.zeros(d), C, n)


def compute_some_values_yay(id_ana):
    # get covariances
    C = get_cov(id_ana)
    if C is None:
        sys.exit('no covariance found!')
    d = C.shape[0]
    condC, w_s, w_c = get_whitening_ops(C)
    X = get_sample(C)
    Xs = sp.dot(X, w_s)
    Xc = sp.dot(X, w_c)
    cov_s = sp.cov(Xs.T)
    cov_c = sp.cov(Xc.T)

    # prints
    print '### norm C ###'
    print 'norm C:', sp.linalg.norm(C)
    print 'cond C:', condC
    print
    print '### norm ###'
    print 'EYE(%s):' % d, sp.sqrt(d)
    print 'SVD     :', sp.linalg.norm(cov_s)
    print 'Cholesky:', sp.linalg.norm(cov_c)

    # return
    return d, sp.linalg.norm(C), sp.linalg.norm(cov_s), sp.linalg.norm(cov_c)

##---MAIN

if __name__ == '__main__':
    # inits
    name = '%d4' % sp.random.random_integers(1000)
    if len(sys.argv) > 1:
        name = str(sys.argv[1])
    q = db.query("""
    SELECT
      a.id
    FROM
      analysis a
    WHERE
      a.id>=1000 AND a.kind='SORT'
    ORDER BY
      a.id
    """)
    q = sp.asanyarray(q)[:, 0].tolist()
    print q

    # archive
    arc = openFile('/home/phil/Data/cov_test_%s' % name, 'w')
    data = sp.zeros((len(q), 6))
    data[:] = sp.nan

    # id loop
    for i, aid in enumerate(q):
        print '###########################################################'
        print 'now doing', aid
        print

        # compute
        data[i, 0] = aid
        try:
            data[i, 1:] = compute_some_values_yay(aid)
        except:
            continue
        finally:
            print

    # save
    arc.createArray(arc.root, 'data', data)

    print 'ALL DONE'
