from typing import List

import os
import argparse
import time
from datetime import datetime

import rpy2.robjects as r_obj
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, pandas2ri, IntVector, StrVector, pandas2ri

import numpy as np
import pandas as pd

import torch

R = r_obj.r

bnlearn = importr('bnlearn')
pcalg = importr('pcalg')

R_SCRIPT_DIR = os.path.dirname(__file__)

R.source(os.path.join(R_SCRIPT_DIR, 'utils.R'))
R.source(os.path.join(R_SCRIPT_DIR, 'generate_data.R'))
R.source(os.path.join(R_SCRIPT_DIR, 'topological_ordering.R'))


NETWORKS = [      # node   edge  parameter    100k    50k
    'sachs',      #   11     17        178     21s    10s
    'child',      #   20     25        230     71s    35s
    'insurance',  #   27     52        984    200s    90s
    'alarm',      #   37     46        509    254s   104s
]
# if n_split is tried then n_split time more runtime is needed
# 100k -> 80k training and 20k validation
#  50k -> 40k training and 10k validation


def bnlearn_data(dataname: str, format: str):
    """

    :param dataname:
    :return: R data.frame each column is a factor
    """
    if dataname in NETWORKS:
        R.set_data(os.path.join(R_SCRIPT_DIR, 'networks', '%s.dsc' % dataname))
        if format == 'R':
            data = R['%s.train' % dataname]
        elif format == 'pandas':
            pandas2ri.activate()
            data = R['%s.train' % dataname]
            pandas2ri.deactivate()
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return data


class TOPORD_CI(object):
    def __init__(self, name: str, alpha: float = 0.05, decay: float = 0.1):
        self.name = 'TOPORD-%s' % name
        self.data = bnlearn_data(dataname=name, format='R')
        # self.marginal_independence_p_values = R.marginal_independence_test(self.data)
        self.permutation_size = R.dim(self.data)[1]
        self.alpha = alpha
        self.decay = decay
        self.valid_ratio = 0.2
        self.n_split = 5
        self.seed = 0
        self.maximize = False

    def __call__(self, x):
        # in $, permutation of lengh N has value between 1 and N while in python 0 and (N - 1)
        p = IntVector(x.numpy().flatten() + 1)
        # score = R.topological_order_score(p, self.data, self.marginal_independence_p_values, self.alpha)[0]
        score = R.topological_order_score_regression(p, self.data, self.decay, self.valid_ratio, self.n_split, self.seed)[0]
        return torch.empty((1, 1), device=x.device).fill_(score) * (-1)  # negative log likelihood


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Permutation Bayesian Optimization')
    parser.add_argument('--data', dest='data', type=str)

    args = parser.parse_args()
    dataname = args.data
    bnlearn_data(dataname, format='pandas')
    evaluator = TOPORD_CI(dataname)
    for _ in range(10):
        x = torch.randperm(evaluator.permutation_size)
        y = evaluator(x)
        print(x, y)
        print('-' * 100)
        print('-' * 100)
        print('-' * 100)
