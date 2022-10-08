import os
import time

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri, IntVector, StrVector, BoolVector

CUR_DIR = os.path.dirname(__file__)

importr('base')
importr('bnlearn')
importr('pcalg')
R = robjects.r
R.source(os.path.join(CUR_DIR, 'utils.R'))


def data_pcalg_format(dataname):
    if dataname == 'sachs':
        data = R['read.table'](os.path.join(CUR_DIR, 'sachs.data.txt'), header=True)
        raise NotImplementedError("Discrete??")
    elif dataname in ['alarm', 'hailfinder']:
        data = R.data_bnlearn2pcalg(R[dataname])
    else:
        raise NotImplementedError
    return data


def structure_learning_pc(dm, nlev, V):
    suff_stat = R['list'](dm=dm, nlev=nlev, adaptDF=False)
    disCItest = R['disCItest']
    cpdag = R.pc(suff_stat, indepTest=disCItest, alpha=0.01, labels=V, verbose=True)
    return cpdag


def parameter_learning():
    pass


if __name__ == '__main__':
    start_time = time.time()
    dm, nlev, V = data_pcalg_format('alarm')
    cpdag = structure_learning_pc(dm, nlev, V)
    print(cpdag)
    print(time.time() - start_time)
