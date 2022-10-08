from typing import List, Union, Dict, Optional

import os
import argparse
import time
from toposort import toposort
from pathos import multiprocessing
import multiprocess.context as ctx
import warnings
from rpy2.rinterface import RRuntimeWarning

import rpy2.robjects as r_obj
from rpy2.robjects.packages import importr
from rpy2.robjects import IntVector, pandas2ri

import numpy as np
import pandas as pd
import scipy.special
import scipy.misc

import torch

ctx._force_start_method('spawn')
warnings.filterwarnings("ignore", category=RRuntimeWarning)

R = r_obj.r

R['options'](warn=-1)

R_SCRIPT_DIR = os.path.dirname(__file__)

R.source(os.path.join(R_SCRIPT_DIR, 'generate_data.R'))
R.source(os.path.join(R_SCRIPT_DIR, 'pc_given_permutation.R'))


NETWORKS = [      # node   edge  parameter    100k    50k
    'sachs',      #   11     17        178     21s    10s
    'child',      #   20     25        230     71s    35s
    'insurance',  #   27     52        984    200s    90s
    'alarm',      #   37     46        509    254s   104s
]
# if n_split is tried then n_split time more runtime is needed
# 100k -> 80k training and 20k validation
#  50k -> 40k training and 10k validation


def bnlearn_data(dataname: str):
    """

    :param dataname:
    :return: R data.frame each column is a factor
    """
    R = r_obj.r
    R['options'](warn=-1)
    R.source(os.path.join(os.path.dirname(__file__), 'generate_data.R'))
    R.set_data(os.path.join(os.path.dirname(__file__), 'networks', '%s.dsc' % dataname))
    pandas2ri.activate()
    data = R['%s.train' % dataname]
    pandas2ri.deactivate()
    return data


def n_sub_forest(adj_mat: pd.DataFrame):
    assert np.all(adj_mat + adj_mat.T < 2)
    parent_count = adj_mat.sum(axis=0)
    have_multiple_parent = parent_count > 1
    return np.prod(parent_count[have_multiple_parent])


def sub_bayesian_forest(adj_mat: pd.DataFrame, ind: int):
    n_nodes = adj_mat.shape[0]
    nodes_name = adj_mat.columns
    parent_count = adj_mat.sum(axis=0)
    have_multiple_parent = parent_count > 1
    n_forests = n_sub_forest(adj_mat)
    assert ind < n_forests
    forest_adj_mat = adj_mat.copy()
    forest_adj_mat[:][:] = 0
    parent_vec = ind2vec(ind=ind, vec_range=parent_count[have_multiple_parent])
    parent_vec_loc = 0
    for i in range(n_nodes):
        colname = nodes_name[i]
        if have_multiple_parent[colname]:
            parent_ind = parent_vec[parent_vec_loc]
            parent = nodes_name[np.where(adj_mat[colname] == 1)[0][parent_ind]]
            forest_adj_mat[colname][parent] = 1
            parent_vec_loc += 1
        else:
            forest_adj_mat[colname] = adj_mat[colname]
    return forest_adj_mat


def log_binomial(n: int, k: Union[int, np.ndarray]):
    if isinstance(k, np.ndarray):
        assert k.ndim == 1
    result = np.sum(np.log(np.arange(1, n + 1)))
    result -= np.cumsum(np.concatenate([np.zeros((1,)), np.log(np.arange(1, np.max(k) + 1))]))[k]
    result -= np.cumsum(np.concatenate([np.zeros((1,)), np.log(np.arange(1, n - np.min(k) + 1))]))[n-k]
    return result


def nml_regret_multinomial(n, k):
    c_matrix = np.ones((k,))
    r1 = np.arange(1, n)
    r2 = n - r1
    log_summand = log_binomial(n=n, k=r1) + r1 * np.log(r1 / n) + r2 * np.log(r2 / n)
    c_matrix[1] = np.exp(scipy.special.logsumexp(log_summand)) + 2
    for i in range(2, k):
        c_matrix[i] = c_matrix[i - 1] + n / (i - 1) * c_matrix[i - 2]
    return np.log(c_matrix[k-1])


def tree_nodes_info(adj_mat: pd.DataFrame):
    n_nodes = adj_mat.shape[0]
    nodes_name = adj_mat.columns
    info_df = pd.DataFrame(data=np.zeros((3, n_nodes), dtype=np.int),
                           columns=nodes_name, index=['depth', 'isRoot', 'isLeaf'])
    for node_name in nodes_name:
        curr_node = node_name
        have_parent = True
        depth = 0
        while have_parent:
            parent_node = nodes_name[np.where(adj_mat[curr_node] == 1)[0]]
            if len(parent_node) == 0:
                have_parent = False
            else:
                curr_node = parent_node[0]
                depth += 1
        info_df.loc['depth', node_name] = depth
    for node_name in nodes_name:
        info_df.loc['isRoot', node_name] = info_df.loc['depth', node_name] == 0
        info_df.loc['isLeaf', node_name] = sum(adj_mat.loc[node_name]) == 0
    return info_df


def test_adjmat(dataname: str):
    R.set_data('networks/%s.dsc' % dataname)
    data = R['%s.train' % dataname]
    graph = R['pc.stable'](data, alpha=0.05)
    graph = R['as.graphAM'](graph)
    adj_mat = R['slot'](graph, 'adjMat')
    pandas2ri.activate()
    adj_mat = R['as.data.frame'](adj_mat)
    pandas2ri.deactivate()
    adj_mat.index = adj_mat.columns
    return adj_mat


def random_direction(adj_mat: pd.DataFrame):
    n_nodes = adj_mat.shape[0]
    nodes_name = adj_mat.columns
    dag_adj_mat = adj_mat.copy()
    for i in range(n_nodes - 1):
        src = nodes_name[i]
        for j in range(i + 1, n_nodes):
            dst = nodes_name[j]
            if adj_mat[src][dst] == adj_mat[dst][src] == 1:
                if np.random.rand() > 0.5:
                    dag_adj_mat[src][dst] = 0
                else:
                    dag_adj_mat[dst][src] = 0
    return dag_adj_mat


def dag_node_depth(adj_mat: pd.DataFrame):
    nodes_name = adj_mat.columns
    parents_ind_dict = {}
    for i in range(adj_mat.shape[1]):
        parents_ind_dict[i] = set(np.where(adj_mat.iloc[:, i] == 1)[0])
    sorted_nodes_ind = toposort(parents_ind_dict)
    sorted_nodes_name = []
    for inds in sorted_nodes_ind:
        sorted_nodes_name.append({nodes_name[elm] for elm in inds})
    return sorted_nodes_name


def ind2vec(ind: int, vec_range: List[int]) -> np.ndarray:
    vec_len = len(vec_range)
    vec = np.zeros((vec_len, ), dtype=np.int)
    divider = np.concatenate([np.ones((1,)), np.cumprod(np.array(vec_range[:-1]))])
    remainder = ind
    for i in range(vec_len - 1, 0, -1):
        vec[i] = remainder // divider[i]
        remainder = remainder % divider[i]
    vec[0] = remainder
    return vec


def vec2ind(vec, vec_range: List[int]) -> int:
    units = np.concatenate([np.ones((1,)), np.cumprod(np.array(vec_range[:-1]))])
    return int(np.sum(vec * units))


def bayesian_network_maximum_log_likelihood(parents_dict: Dict, topologically_ordered_nodes_name: List[set],
                                            data: pd.DataFrame, data_cats: Dict[str, pd.Index]):
    n_data = data.shape[0]
    log_likelihood = 0
    for nodes_name_set in topologically_ordered_nodes_name:
        for node_name in nodes_name_set:
            parents = sorted(list(parents_dict[node_name]))
            if len(parents) == 0:
                sub_data = data.loc[:, [node_name] + parents]
                sub_data_count = sub_data.value_counts()
                log_likelihood += np.sum(np.log(sub_data_count / n_data) * sub_data_count)
            else:
                sub_data_parent_code = data.loc[:, parents].copy()
                for col in parents:
                    sub_data_parent_code.loc[:, col] = data.loc[:, col].cat.codes
                parents_code_range = [len(data[elm].cat.categories) for elm in parents]
                multiplier = np.concatenate([np.ones((1,)), np.cumprod(parents_code_range[:-1])]).astype(np.int)
                parents_config_id = (sub_data_parent_code * multiplier.reshape((1, -1))).sum(axis=1)
                sub_data_code = pd.concat([data[node_name], parents_config_id], axis=1)
                sub_data_code.columns = [node_name, 'parent_config_id']
                sub_data_count = sub_data_code.value_counts()
                sub_data_count_denom = sub_data_count.sum(level='parent_config_id').dropna()
                log_likelihood += \
                    sum([np.sum(np.log(sub_data_count[c] / sub_data_count_denom[sub_data_count[c].index])
                                * sub_data_count[c]) for c in data_cats[node_name] if c in sub_data_count])
    return log_likelihood


def bayesian_network_maximum_log_likelihood_wrapper(adj_mat: pd.DataFrame, data: pd.DataFrame):
    nodes_name = adj_mat.columns
    topologically_ordered_nodes_name = dag_node_depth(adj_mat)
    data_cats = {node_name: data[node_name].cat.categories for node_name in nodes_name}
    parents_dict = dict()
    for node_name in nodes_name:
        parents = sorted(nodes_name[np.where(adj_mat.loc[:, node_name] == 1)[0]])
        parents_dict[node_name] = parents
    log_lik = bayesian_network_maximum_log_likelihood(
        parents_dict=parents_dict, topologically_ordered_nodes_name=topologically_ordered_nodes_name,
        data=data, data_cats=data_cats)
    return log_lik


def datasampling(nodes_name, data_cats: Dict[str, pd.Index], n_data: int):
    series_list = []
    for node_name in nodes_name:
        series = pd.Series(data=np.random.choice(data_cats[node_name], size=(n_data, )),
                           name=node_name, dtype="category")
        series.cat.categories = data_cats[node_name]
        series_list.append(series)
    df = pd.concat(series_list, axis=1).loc[:, nodes_name]
    return df


def nml_regret_estimate(data_cats: Dict, n_samples: int, n_data: int, adj_mat: pd.DataFrame,
                        sample_size_correction: bool = True):
    nodes_name = adj_mat.columns
    topologically_ordered_nodes_name = dag_node_depth(adj_mat)
    parents_dict = dict()
    for node_name in nodes_name:
        parents = sorted(nodes_name[np.where(adj_mat.loc[:, node_name] == 1)[0]])
        parents_dict[node_name] = parents
    log_lik_mat = np.zeros((n_samples))
    for s in range(n_samples):
        sample = datasampling(nodes_name=nodes_name, data_cats=data_cats, n_data=n_data)
        log_lik = bayesian_network_maximum_log_likelihood(
            parents_dict=parents_dict, topologically_ordered_nodes_name=topologically_ordered_nodes_name,
            data=sample, data_cats=data_cats)
        log_lik_mat[s] = log_lik
    estimate = scipy.special.logsumexp(log_lik_mat)
    if sample_size_correction:
        estimate += n_data * np.sum(np.log([len(elm) for elm in data_cats.values()])) - np.log(n_data)
    return estimate


def nml_regret_estimate_parallel(
        data_cats: Dict, n_samples: int, n_data: int, adj_mat: pd.DataFrame, n_processes: int):

    pool = multiprocessing.Pool(n_processes)
    if n_samples % n_processes == 0:
        args_list = [(n_samples // n_processes, ) for _ in range(n_processes)]
    else:
        args_list = [(n_samples // n_processes, ) for _ in range(n_processes - 1)] + [(n_samples % n_processes, )]

    def nml_regret_estimate_wrapper(s):
        return nml_regret_estimate(data_cats=data_cats, n_samples=s, n_data=n_data, adj_mat=adj_mat,
                                   sample_size_correction=False)

    regret_estimate_list = pool.starmap_async(nml_regret_estimate_wrapper, args_list).get()
    sample_size_correction = n_data * np.sum(np.log([len(elm) for elm in data_cats.values()])) - np.log(n_samples)
    return scipy.special.logsumexp(regret_estimate_list) + sample_size_correction


def random_adj_mat(n_nodes: int, n_edges: int, nodes_name, seed: Optional[int] = None):
    seed1, seed2 = np.random.RandomState(seed).randint(0, 10000, (2, ))
    p = np.random.RandomState(seed1).permutation(n_nodes)
    max_edges = int(n_nodes * (n_nodes - 1) / 2)
    have_edge_flatten = np.concatenate([np.zeros((n_edges,)), np.ones((max_edges - n_edges,))])
    have_edge = np.zeros((n_nodes, n_nodes), dtype=np.int)
    cnt = 0
    for i in range(n_nodes - 1):
        have_edge[i, i + 1:] = have_edge_flatten[cnt:cnt+n_nodes - i - 1]
        cnt += n_nodes - i - 1
    adj_mat = np.zeros((n_nodes, n_nodes), dtype=np.int)
    for i in range(n_nodes - 1):
        for j in range(i + 1, n_nodes):
            if have_edge[i, j]:
                adj_mat[p[i], p[j]] = 1
    return pd.DataFrame(data=adj_mat, columns=nodes_name, index=nodes_name)


def pc_given_permutation(data: pd.DataFrame, permutation: np.ndarray):
    R = r_obj.r
    R['options'](warn=-1)
    R.source(os.path.join(os.path.dirname(__file__), 'pc_given_permutation.R'))
    pandas2ri.activate()
    adj_mat = R.pc_given_permutation(data, IntVector(permutation + 1))  # indexing beginning number difference
    pandas2ri.deactivate()
    return adj_mat


class DAGNML(object):
    def __init__(self, name: str):
        self.name = 'DAGNML-%s' % name
        self.data = bnlearn_data(dataname=name)
        self.maximize = False
        self.filename = __file__
        self.n_data, self.n_nodes = self.data.shape
        self.n_samples = 10000
        self.nodes_name = self.data.columns
        self.data_cats = {node_name: self.data[node_name].cat.categories for node_name in self.nodes_name}
        self.n_processes = multiprocessing.cpu_count() - 2

    def __call__(self, x):
        permutation = x.numpy().flatten()
        adj_mat = pc_given_permutation(data=self.data, permutation=permutation)
        maximum_log_likelihood = bayesian_network_maximum_log_likelihood_wrapper(adj_mat=adj_mat, data=self.data)
        regret_estimate = nml_regret_estimate_parallel(
            data_cats=self.data_cats, n_samples=self.n_samples, n_data=self.n_data, adj_mat=adj_mat,
            n_processes=self.n_processes)
        return torch.empty((1, 1), device=x.device).fill_(-maximum_log_likelihood + regret_estimate)


def test_evaluation(seed: Optional[int] = None, dataname: str = 'sachs', n_repeat: int = 10, n_sample: int = 1000,
                    n_processes: int = multiprocessing.cpu_count() - 2):
    seed1, seed2 = (None, None) if seed is None else np.random.RandomState(seed).randint(0, 10000, (2,))
    print('-' * 100)
    data = bnlearn_data(dataname)
    n_data = data.shape[0]
    n_nodes = data.shape[1]
    nodes_name = data.columns
    max_edges = n_nodes * (n_nodes - 1) / 2
    expected_edges = np.random.RandomState(seed1).randint(n_nodes, n_nodes * 3)
    adj_mat = random_adj_mat(n_nodes=n_nodes, expected_edges=expected_edges, nodes_name=nodes_name, seed=seed2)
    data_cats = {node_name: data[node_name].cat.categories for node_name in nodes_name}
    estimate_list = []
    sample_size_correction_ = n_data * np.sum(np.log(
        [len(elm_) for elm_ in data_cats.values()])) - np.log(n_sample)
    print('Seed : %4d -> (%4d,%4d),  Edges : %3d/%3d ,  Data : %s,  %d processes in parallel' %
          (seed, seed1, seed2, expected_edges, max_edges, dataname, n_processes))
    print('%d repetition with %d samples %6d data points (sample size correction : %+12.4f)'
          % (n_repeat, n_sample, n_data, sample_size_correction_))
    bn_mll_start_time = time.time()
    bn_mll = bayesian_network_maximum_log_likelihood_wrapper(adj_mat=adj_mat, data=data)
    print('maximum log likelihood : %+12.4f (%.6f seconds)' % (bn_mll, time.time() - bn_mll_start_time))
    elapsed_time_list = []
    for r_ in range(n_repeat):
        start_time = time.time()
        if n_processes == 1:
            estimate = nml_regret_estimate(data_cats=data_cats, n_samples=n_sample, n_data=n_data, adj_mat=adj_mat)
        else:
            estimate = nml_regret_estimate_parallel(data_cats=data_cats, n_samples=n_sample, n_data=n_data,
                                                    adj_mat=adj_mat, n_processes=n_processes)
        estimate_list.append(estimate)
        elapsed_time_list.append(time.time() - start_time)
        print('%4d run - %6d seconds: regret %+12.4f' % (r_, int(elapsed_time_list[-1]), estimate))
    print('-' * 100)
    # print(np.mean(estimate_list))
    # print(np.std(estimate_list))
    # print(sum(elapsed_time_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Permutation Bayesian Optimization')
    parser.add_argument('--data', dest='data', type=str, default=None)
    parser.add_argument('--seed', dest='seed', type=int, default=None)
    parser.add_argument('--sample', dest='sample', type=int, default=None)

    args = parser.parse_args()
    if args.seed is None:
        for d_ in ['insurance', 'child', 'sachs']:
            for s_ in range(4):
                for n_ in [10000, 100000]:
                    test_evaluation(seed=s_, dataname=d_, n_repeat=2, n_sample=n_)
    else:
        test_evaluation(seed=args.seed, dataname=args.data, n_repeat=2, n_sample=args.sample)

