from typing import List, Dict, Tuple

import math
import toposort
from pathos import multiprocessing
import multiprocess.context as ctx
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

from LAW2ORDER.experiments.data.causal_discovery.normalized_marginal_likelihood import bnlearn_data, random_adj_mat


ctx._force_start_method('spawn')


class DAGOperationType(Enum):
    Addition = 'Addition'
    Removal = 'Removal'
    Reversal = 'Reversal'


def regret_approximation_multinomial(n_data: int, n_cats: int):
    alpha = n_cats / n_data
    c_alpha = 0.5 * (1 + (1 + 4 / alpha) ** 0.5)
    regret = (n_data * (math.log(alpha) + (alpha + 2) * math.log(c_alpha) - 1 / c_alpha)
              - 0.5 * math.log(c_alpha + 2 / alpha))
    return regret


def log_onedim_nml(chosen_nodes_name: List[str], data: pd.DataFrame):
    sub_data_code = data.loc[:, chosen_nodes_name].copy()
    for col in chosen_nodes_name:
        sub_data_code.loc[:, col] = data.loc[:, col].cat.codes
    code_range = [len(data[elm].cat.categories) for elm in chosen_nodes_name]
    multiplier = np.concatenate([np.ones((1,)), np.cumprod(code_range[:-1])]).astype(np.int)
    sub_data_id = (sub_data_code * multiplier.reshape((1, -1))).sum(axis=1)
    sub_data_count = sub_data_id.value_counts()
    sub_data_count_nonzero = sub_data_count[sub_data_count != 0]
    log_likelihood = np.sum(np.log(sub_data_count_nonzero) * sub_data_count_nonzero)
    onedim_n_cats = int(np.prod(code_range))
    n_data = data.shape[0]
    regret = regret_approximation_multinomial(n_data=n_data, n_cats=onedim_n_cats)
    return log_likelihood - regret


def node_qnml(node_name: str, data: pd.DataFrame, adj_mat: pd.DataFrame):
    parents_nodes_name = list(data.columns[np.where(adj_mat.loc[:, node_name] == 1)[0]])
    numerator_chosen_nodes_name = [node_name] + parents_nodes_name
    qnml_numerator = log_onedim_nml(chosen_nodes_name=numerator_chosen_nodes_name, data=data)
    qnml = qnml_numerator
    if len(parents_nodes_name) > 0:
        qnml_denomenator = log_onedim_nml(chosen_nodes_name=parents_nodes_name, data=data)
        qnml = qnml - qnml_denomenator
    return qnml


def adj_mat2dict(adj_mat: pd.DataFrame):
    adj_mat_dict = dict()
    for i in range(adj_mat.shape[1]):
        adj_mat_dict[i] = set(np.where(adj_mat.iloc[:, i])[0])
    return adj_mat_dict


def adj_dict2mat(adj_dict: Dict, nodes_name):
    n_nodes = len(nodes_name)
    adj_mat = pd.DataFrame(data=np.zeros((n_nodes, n_nodes), dtype=int), columns=nodes_name, index=nodes_name)
    for node_ind, parents in adj_dict.items():
        for p_ind in parents:
            adj_mat.iloc[p_ind, node_ind] = 1
    return adj_mat


def acyclic_neighbor_moves(adj_mat: pd.DataFrame):
    neighbor_list = []
    nodes_name = list(adj_mat.columns)
    for src in nodes_name:
        for dst in nodes_name:
            if src != dst:
                if adj_mat.loc[src, dst] == 0:  # edge addition
                    if adj_mat.loc[dst, src] == 0:  # without this condition being cyclic with undirected edge
                        cand_adj_mat = adj_mat.copy()
                        cand_adj_mat.loc[src, dst] = 1
                        try:
                            list(toposort.toposort(data=adj_mat2dict(cand_adj_mat)))
                            neighbor_list.append((src, dst, DAGOperationType.Addition.value))
                        except toposort.CircularDependencyError:
                            pass
                else:  # adj_mat.loc[src, dst] == 1
                    # edge removal
                    cand_adj_mat = adj_mat.copy()
                    cand_adj_mat.loc[src, dst] = 0
                    try:
                        list(toposort.toposort(data=adj_mat2dict(cand_adj_mat)))
                        neighbor_list.append((src, dst, DAGOperationType.Removal.value))
                    except toposort.CircularDependencyError:
                        pass
                    # edge reversal
                    cand_adj_mat = adj_mat.copy()
                    cand_adj_mat.loc[src, dst] = 0
                    cand_adj_mat.loc[dst, src] = 1
                    try:
                        list(toposort.toposort(data=adj_mat2dict(cand_adj_mat)))
                        neighbor_list.append((src, dst, DAGOperationType.Reversal.value))
                    except toposort.CircularDependencyError:
                        pass
    return neighbor_list


def best_neighbor(adj_mat: pd.DataFrame, node_qnml_dict: Dict, data: pd.DataFrame) -> Tuple:
    moves_to_neighbor = acyclic_neighbor_moves(adj_mat=adj_mat)
    score_original = sum(list(node_qnml_dict.values()))
    best_score = score_original
    best_neighbor = adj_mat
    best_op_src = None
    best_op_dst = None
    best_op_type = None
    best_op_scr_score = None
    best_op_dst_score = None
    for src, dst, op_type in moves_to_neighbor:
        src_score_original = node_qnml_dict[src]
        dst_score_original = node_qnml_dict[dst]
        neighbor_adj_mat = adj_mat.copy()
        if op_type == DAGOperationType.Addition.value:
            neighbor_adj_mat.loc[src, dst] = 1
        elif op_type == DAGOperationType.Removal.value:
            neighbor_adj_mat.loc[src, dst] = 0
        elif op_type == DAGOperationType.Reversal.value:
            neighbor_adj_mat.loc[src, dst] = 0
            neighbor_adj_mat.loc[dst, src] = 1
        src_score_neighbor = node_qnml(node_name=src, data=data, adj_mat=neighbor_adj_mat)
        dst_score_neighbor = node_qnml(node_name=dst, data=data, adj_mat=neighbor_adj_mat)
        score_neighbor = (score_original
                          - src_score_original - dst_score_original + src_score_neighbor + dst_score_neighbor)
        if score_neighbor >= best_score:
            best_score = score_neighbor
            best_neighbor = neighbor_adj_mat.copy()
            best_op_src = src
            best_op_dst = dst
            best_op_type = op_type
            best_op_scr_score = src_score_neighbor
            best_op_dst_score = dst_score_neighbor
    updated_node_qnml_dict = node_qnml_dict.copy()
    updated_node_qnml_dict[best_op_src] = best_op_scr_score
    updated_node_qnml_dict[best_op_dst] = best_op_dst_score
    return best_neighbor, best_score, best_op_src, best_op_dst, best_op_type, updated_node_qnml_dict


def score_maximization(init_adj_mat: pd.DataFrame, data: pd.DataFrame, verbose: bool = False):
    nodes_name = data.columns
    node_qnml_dict = {node_name: node_qnml(node_name=node_name, data=data, adj_mat=init_adj_mat)
                      for node_name in nodes_name}
    prev_score = sum(list(node_qnml_dict.values()))
    prev_adj_mat = init_adj_mat.copy()
    cnt = 0
    if verbose:
        print('%s Score Maximization has started with %+14.6f' % (datetime.now().strftime('%H:%M:%S'), prev_score))
    while True:
        adj_mat, score, _, _, _, node_qnml_dict = best_neighbor(
            adj_mat=prev_adj_mat, node_qnml_dict=node_qnml_dict, data=data)
        cnt += 1
        if score > prev_score:
            prev_adj_mat = adj_mat
            prev_score = score
        else:
            break
        if cnt % 5 == 0 and verbose:
            print('%s %8d updates : %+14.6f' % (datetime.now().strftime('%H:%M:%S'), cnt, score))
    if verbose:
        print('%s %8d updates : %+14.6f' % (datetime.now().strftime('%H:%M:%S'), cnt, prev_score))
    return prev_adj_mat, prev_score


def qnml_structure_learning(data: pd.DataFrame, verbose: bool = False):
    n_nodes = data.shape[1]
    nodes_name = data.columns
    while True:
        try:
            n_edges = np.random.randint(2, int(n_nodes * (n_nodes - 1) / 2 * 0.25))
            adj_mat = random_adj_mat(n_nodes=n_nodes, n_edges=n_edges, nodes_name=nodes_name, seed=None)
            adj_dict = adj_mat2dict(adj_mat)
            list(toposort.toposort(adj_dict))
            break
        except toposort.CircularDependencyError:
            pass
    optimized_adj_mat, optimized_score = score_maximization(init_adj_mat=adj_mat, data=data, verbose=verbose)
    return optimized_adj_mat, optimized_score


def qnml_structure_learning_in_parallel(dataname: str, n_random_inits: int, n_processes: int):
    data = bnlearn_data(dataname)

    pool = multiprocessing.Pool(n_processes)
    args_list = [(data, False) for _ in range(n_random_inits)]
    adj_mat_list, score_list = list(zip(*pool.starmap_async(qnml_structure_learning, args_list).get()))
    best_ind = np.argmax(score_list)
    best_adj_mat = adj_mat_list[best_ind]
    best_score = score_list[best_ind]
    return best_adj_mat, best_score


def test_qnml(dataname: str, seed: int):
    seed1, seed2 = (None, None) if seed is None else np.random.RandomState(seed).randint(0, 10000, (2,))
    print('-' * 100)
    data = bnlearn_data(dataname)
    n_data = data.shape[0]
    n_nodes = data.shape[1]
    nodes_name = data.columns
    max_edges = n_nodes * (n_nodes - 1) / 2
    while True:
        try:
            expected_edges = np.random.RandomState(None).randint(n_nodes, n_nodes * 3)
            adj_mat = random_adj_mat(n_nodes=n_nodes, expected_edges=expected_edges, nodes_name=nodes_name, seed=None)
            adj_dict = adj_mat2dict(adj_mat)
            list(toposort.toposort(adj_dict))
            break
        except toposort.CircularDependencyError:
            pass
    optimized_adj_mat = score_maximization(init_adj_mat=adj_mat, data=data)



if __name__ == '__main__':
    test_qnml(dataname='sachs', seed=0)