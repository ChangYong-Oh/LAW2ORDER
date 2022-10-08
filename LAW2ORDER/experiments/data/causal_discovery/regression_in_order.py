from typing import List, Tuple, Optional

import os
import time
import argparse
from pathos import multiprocessing
import multiprocess.context as ctx

import rpy2.robjects as r_obj
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, IntVector

import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

import torch
from torch import Tensor
from torch import nn
from torch.optim import LBFGS
from torch.optim.lr_scheduler import ExponentialLR

ctx._force_start_method('spawn')

r_obj.r['options'](warn=-1)
bnlearn = importr('bnlearn')
#pcalg = importr('pcalg')

R_SCRIPT_DIR = os.path.dirname(__file__)
R = r_obj.r
R.source(os.path.join(R_SCRIPT_DIR, 'generate_data.R'))


def load_R_data(dataname: str):
    R.set_data(os.path.join(R_SCRIPT_DIR, 'networks', '%s.dsc' % dataname))
    pandas2ri.activate()
    data = R['%s.train' % dataname]
    pandas2ri.deactivate()
    return data


def onehot_tensor(data: pd.DataFrame, regressors: List[str], response: Optional[str] = None) \
        -> Tuple[Optional[Tensor], Tensor, List[int]]:
    data_colnames = data.columns
    if response is not None:
        assert response in data_colnames
        data_y_onehot = torch.from_numpy(pd.get_dummies(data[response]).to_numpy(dtype=np.float32))
    else:
        data_y_onehot = None
    assert np.all([elm in data_colnames for elm in regressors])
    data_x_onehot_list = [pd.get_dummies(data[elm], prefix=elm) for elm in regressors]
    data_x_group_size = [elm.shape[1] for elm in data_x_onehot_list]
    data_x_onehot = torch.from_numpy(pd.concat(data_x_onehot_list, axis=1).to_numpy(dtype=np.float32))
    return data_y_onehot, data_x_onehot, data_x_group_size


def predict(x: pd.DataFrame, weight: Tensor, bias: Optional[Tensor] = None,
            regressors: Optional[List[str]] = None, format: str = 'softmax'):
    x_colnames = x.columns
    if regressors is not None:
        assert np.all([regressors[elm] == x_colnames[elm] for elm in range(len(regressors))])
    _, input_onehot, _ = onehot_tensor(data=x, regressors=regressors)
    softmax = torch.softmax(torch.mm(input_onehot, weight) + (bias if bias is not None else 0), dim=1)
    if format == 'softmax':
        return softmax
    elif format == 'onehot':
        return (softmax == torch.max(softmax, dim=1, keepdim=True)[0]).int()


def sparse_group_lasso(weight: Tensor, group_size: List[int], alpha: float) -> Tensor:
    """
    sparse group lasso regularizer is adopted to multinomial logistic regression
    A SPARSE-GROUP LASSO; NOAH SIMON, JEROME FRIEDMAN, TREVOR HASTIE,AND ROB TIBSHIRANI
    :param weight:
    :param group_size:
    :param alpha: 0 -> group lasso, 1 -> lasso
    :return:
    """
    l1 = torch.sum(torch.abs(weight)) / weight.size()[1] ** 0.5  # normalize by the dimension of the output
    group_boundary = [0] + list(np.cumsum(group_size))
    group_l2_list = \
        [(torch.sum(weight[group_boundary[elm]:group_boundary[elm+1]] ** 2).view(1, 1)
          * (group_boundary[elm+1] - group_boundary[elm])) ** 0.5 for elm in range(len(group_size))]
    group_l2 = torch.sum(torch.cat(group_l2_list))
    return (1 - alpha) * group_l2 + alpha * l1


def sparsify_weight(weight: Tensor, group_size: List[int], threshold: float):
    if len(group_size) == 1:
        return weight
    group_boundary = [0] + list(np.cumsum(group_size))
    group_ssq = torch.cat([torch.sum(weight[group_boundary[elm]:group_boundary[elm+1]] ** 2).view(1, 1)
                           for elm in range(len(group_size))]).view(-1)
    ordered_group_ssq, original_loc = torch.sort(group_ssq, dim=0)
    accum_signal = torch.cumsum(ordered_group_ssq, dim=0)
    n_cut = torch.sum(accum_signal < threshold * accum_signal[-1])
    group_to_cut = original_loc[:n_cut]
    sparsified_weight = weight.clone()
    for g in group_to_cut:
        sparsified_weight[group_boundary[g]:group_boundary[g+1]] = 0
    # print(accum_signal / accum_signal[-1])
    # print('Zeros %d/%d' % (torch.sum(sparsified_weight == 0).int(), sparsified_weight.numel()))
    return sparsified_weight


def uncertainty_coefficient(confusion_matrix: np.ndarray, symmetric: bool = True):
    """

    :param confusion_matrix: prediction \ ground truth
    :param symmetric:
    :return:
    """
    n_total = confusion_matrix.sum()
    p = confusion_matrix / n_total
    p_pred = confusion_matrix.sum(axis=1, keepdims=True) / n_total
    p_gt = confusion_matrix.sum(axis=0, keepdims=True) / n_total
    p_pred_gt = p / p_gt
    p_gt_pred = p / p_pred
    h_pred = -np.sum(p_pred * np.log(p_pred))
    h_gt = -np.sum(p_gt * np.log(p_gt))
    h_pred_gt = -np.sum(p * np.log(p_pred_gt))
    h_gt_pred = -np.sum(p * np.log(p_gt_pred))
    u_pred_gt = (h_pred - h_pred_gt) / h_pred
    u_gt_pred = (h_gt - h_gt_pred) / h_gt
    if symmetric:
        return (h_pred * u_pred_gt + h_gt * u_gt_pred) / (h_pred + h_gt)
    else:
        return u_pred_gt


class InfoProcessor(object):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.line_list = []

    def process(self, line: str):
        if self.verbose:
            print(line)
        self.line_list.append(line)


def info_line_processsing(info_str_list: List[str], info_line: str, verbose: bool) -> List[str]:
    if verbose:
        print(info_line)
    info_str_list.append(info_line)
    return info_str_list


def regression_given_permutation(train_data: pd.DataFrame, valid_data: pd.DataFrame,
                                 permutation: np.ndarray, alpha=0.5, lamda=1e-2, threshold=0.05, n_init=5,
                                 verbose: bool = False):
    """

    :param train_data:
    :param valid_data:
    :param permutation:
    :param alpha: weighting between LASSO regularizer and group LASSO regularizer 0 -> group lasso, 1 -> lasso
    :param lamda: intensity of the regularizer(sparsified group LASSO)
    :param threshold: criterion used to zero out the part of the trained weight with small values
    :param n_init: each regression is fitted with n_init random initialization
    :param verbose:
    :return:
    """
    colnames = train_data.columns
    p_size = permutation.size
    adj_mat = pd.DataFrame(np.zeros((p_size, p_size)).astype(int), columns=colnames, index=colnames)
    valid_loglik_sum = 0
    info = InfoProcessor(verbose=verbose)
    for p in range(1, p_size):
        response = colnames[permutation[p]]
        regressors = colnames[permutation[:p]]
        train_y_onehot, _, group_size = onehot_tensor(data=train_data, regressors=regressors, response=response)
        valid_y_onehot, _, _ = onehot_tensor(data=valid_data, regressors=regressors, response=response)

        train_loss_list = []

        best_valid_pred = None
        best_valid_xentropy = 0
        best_valid_acc = None
        best_weight = None
        best_sparsified_weight = None
        best_bias = None
        for _ in range(n_init):
            weight, bias, train_loss = fit_cat2cat_regression(
                data=train_data, response=response, regressors=regressors, alpha=alpha, decay=lamda)

            sparsified_weight = sparsify_weight(weight=weight, group_size=group_size, threshold=threshold)
            valid_softmax = predict(x=valid_data[regressors], weight=sparsified_weight, bias=bias,
                                    regressors=regressors, format='softmax')
            valid_xentropy = torch.mean(torch.sum(valid_softmax * valid_y_onehot, dim=1))
            valid_pred = predict(x=valid_data[regressors], weight=sparsified_weight, bias=bias,
                                 regressors=regressors, format='onehot')
            valid_acc = torch.mean(torch.all(torch.eq(valid_y_onehot, valid_pred), dim=1).float())
            train_loss_list.append(train_loss)

            if best_valid_xentropy < valid_xentropy:
                best_valid_xentropy = valid_xentropy
                best_valid_pred = valid_pred
                best_valid_acc = valid_acc * 100
                best_weight = weight.clone()
                best_sparsified_weight = sparsified_weight.clone()
                best_bias = bias.clone()
        valid_loglik_sum += -best_valid_xentropy.item()
        valid_pred1 = predict(x=valid_data[regressors], weight=best_sparsified_weight, bias=best_bias,
                              regressors=regressors, format='onehot')
        valid_acc1 = torch.mean(torch.all(torch.eq(valid_y_onehot, valid_pred1), dim=1).float()) * 100
        valid_pred2 = predict(x=valid_data[regressors], weight=best_weight, bias=best_bias,
                              regressors=regressors, format='onehot')
        valid_acc2 = torch.mean(torch.all(torch.eq(valid_y_onehot, valid_pred2), dim=1).float()) * 100
        info.process('>' * 100)
        group_boundary = [0] + list(np.cumsum(group_size))
        group_ssq = \
            torch.cat([torch.sum(best_sparsified_weight[group_boundary[elm]:group_boundary[elm + 1]] ** 2).view(1, 1)
                       for elm in range(len(group_size))]).view(-1)
        sparsified_regressors = list(np.array(regressors)[(group_ssq > 0).numpy()])
        info.process('Active variables : %d out of %d as below' % (torch.sum(group_ssq > 0).int(), group_ssq.numel()),)
        info.process('Validation accuracy : sparsified weight %7.2f%% / original weight %7.2f%%' %
                     (valid_acc1.item(), valid_acc2.item()))

        n_cats = valid_y_onehot.size()[1]
        cat_values = train_data[response].cat.categories.to_list()
        info.process(('Original   %s ~ ' % response) + ' + '.join(['%s' % elm for elm in regressors]))
        info.process(('Sparsified %s ~ ' % response) + ' + '.join(['%s' % elm for elm in sparsified_regressors]))
        confusion_matrix = pd.DataFrame(data=np.zeros((n_cats, n_cats), dtype=np.int),
                                        columns=cat_values, index=cat_values)
        for gt_ind, gt in enumerate(cat_values):
            for pred_ind, pred in enumerate(cat_values):
                confusion_matrix[gt][pred] = \
                    torch.sum(valid_y_onehot[:, gt_ind] + best_valid_pred[:, pred_ind] == 2).item()
        info.process('Prediction \ Ground Truth')
        info.process(confusion_matrix.to_string())
        # 1 is added to avoid zero element error with the smallest perturbation
        pred_gt_independence_test_p_value = chi2_contingency(confusion_matrix.to_numpy() + 1)[1]
        theils_u = uncertainty_coefficient(confusion_matrix.to_numpy() + 1)
        pred_gt_independent = theils_u < 0.2
        info.process('Symmetrized uncertainty coefficient [-1, 1] : %+.4f' % theils_u)
        info.process('p-value : %.4f thus' % pred_gt_independence_test_p_value)
        info.process('Independent' if pred_gt_independent else 'Dependent')
        if not pred_gt_independent:
            for sr in sparsified_regressors:
                adj_mat[response][sr] = 1
    info.process('-' * 100)
    info.process('alpha : %5.2f / lambda %.2E / threshold %.2f / %d random inits' %
                      (alpha, lamda, threshold, n_init))
    info.process('validation log likelihood sum %+.6f' % valid_loglik_sum)
    info.process('#' * 100)
    info.process(adj_mat.to_string())
    info.process('%d directed edges' % adj_mat.sum().sum())
    info.process(' '.join([colnames[permutation[elm]] for elm in range(p_size)]))
    info.process('#' * 100)

    return valid_loglik_sum, adj_mat, '\n'.join(info.line_list)


def fit_cat2cat_regression(data: pd.DataFrame, response: str, regressors: List[str], decay: float, alpha: float):
    """
    To encourage variable-wise sparsity with one-hot encoding, group LASSO is used.
    :param data
    :param response:
    :param regressors:
    :return:
    """
    y_onehot, x_onehot, data_x_group_size = onehot_tensor(data=data, regressors=regressors, response=response)

    weight = x_onehot.new_zeros(x_onehot.size()[1], y_onehot.size()[1]).requires_grad_()
    bias = x_onehot.new_zeros(1, y_onehot.size()[1]).requires_grad_()
    nn.init.xavier_normal_(weight)

    optimizer = LBFGS(params=[weight], lr=0.1)
    # scheduler = ExponentialLR(optimizer, gamma=0.95)

    best_loss = np.inf
    prev_loss = np.inf
    best_weight = weight
    for i in range(100):
        def closure():
            optimizer.zero_grad()
            output = torch.softmax(torch.mm(x_onehot, weight) + bias, dim=1)
            xentropy = torch.mean(torch.sum(output * y_onehot, dim=1))
            reg = sparse_group_lasso(weight=weight, group_size=data_x_group_size, alpha=alpha)
            loss = -xentropy + decay * reg
            loss.backward()
            # print('             LOSS : %+12.8f, FIT : %+12.8f, REG : %+12.8f'
            #       % (loss.item(), xentropy.item(), reg.item()))
            return loss
        optimizer.step(closure)
        # scheduler.step()
        loss = optimizer.state_dict()['state'][0]['prev_loss']
        if loss < best_loss:
            best_weight = weight.clone()
            best_loss = loss
        # print('%4d step - FROM : %+12.8f, TO : %+12.8f, Dec.by : %+12.8f' %
        #       ((i + 1), prev_loss, loss, prev_loss - loss))
        if (np.isinf(loss) or np.isnan(loss)
                or (prev_loss - loss) / max(1, abs(prev_loss), abs(loss)) <= 2.220446049250313e-09):
            weight = best_weight
            loss = best_loss
            break
        prev_loss = loss

    return weight.detach(), bias.detach(), loss


def cross_valiation_cat2cat_regression(permutation: np.ndarray, data: pd.DataFrame, valid_ratio: float = 0.2,
                                       alpha: float = 0.5, lamda: float = 1e-2, threshold: float = 0.1,
                                       n_init: int = 5, parallel: bool = True):
    """
    this is not rigorously cross validation but it uses different random splits
    most of parameters are explained in regression_given_permutation
    :param permutation:
    :param data:
    :param valid_ratio:
    :param alpha:
    :param lamda:
    :param threshold:
    :param n_init:
    :param parallel: each run of regression_given_permutation uses 10~12 cores, on a machine with 32 cores,
    running 3 processes in parallel makes the execution way slower,
    it takes almost the same as running 2 runs of 2 processes, which gives the average over 4 splits.
    :return:
    """
    # To guarantee that each evaluation of the permutation uses the same data split, assuming that n_split = 5
    n_split = 2
    split_seed_list = [3190, 3439]  # generated by np.random.RandomState(3457).randint(1000, 10000, (4, ))
    assert len(split_seed_list) == n_split

    colnames = data.columns
    n_data = data.shape[0]
    n_train = int(n_data * (1 - valid_ratio))

    if not parallel:
        valid_loglik_list = []
        adj_mat_list = []
        info_str_list = []
        for split_seed in split_seed_list:
            shuffled_ind = np.random.RandomState(split_seed).permutation(n_data)
            train_data = data.iloc[shuffled_ind[:n_train]]
            valid_data = data.iloc[shuffled_ind[n_train:]]
            valid_loglik, adj_mat, info_str = regression_given_permutation(
                train_data=train_data, valid_data=valid_data, permutation=permutation,
                alpha=alpha, lamda=lamda, threshold=threshold, n_init=n_init, verbose=True)
            valid_loglik_list.append(valid_loglik)
            adj_mat_list.append(adj_mat)
            info_str_list.append(info_str)
    else:
        pool = multiprocessing.Pool(max(1, min(multiprocessing.cpu_count() // 12, n_split)))
        args_list = []
        for i in range(n_split):
            shuffled_ind = np.random.RandomState(split_seed_list[i]).permutation(n_data)
            train_data = data.iloc[shuffled_ind[:n_train]]
            valid_data = data.iloc[shuffled_ind[n_train:]]
            args_list.append((train_data, valid_data, permutation, alpha, lamda, threshold, n_init))
        valid_loglik_list, adj_mat_list, info_str_list = \
            list(zip(*pool.starmap_async(regression_given_permutation, args_list).get()))

    edge_frequency = adj_mat_list[0][colnames].loc[colnames]
    for am in range(1, n_split):
        edge_frequency += adj_mat_list[am][colnames].loc[colnames]
    edge_frequency /= n_split
    mean_valid_loglik = float(np.mean(valid_loglik_list))

    max_name_len = max([len(elm) for elm in colnames])
    fmt_str_d = '%%%dd' % max_name_len
    fmt_str_s = '%%%ds' % max_name_len

    print('with alpha : %5.2f / lambda %.2E / threshold %.2f / %d random inits / %d split' %
          (alpha, lamda, threshold, n_init, n_split))
    print(' '.join([fmt_str_d % permutation[elm].item() for elm in range(len(colnames))]))
    print(' '.join([fmt_str_s % colnames[permutation[elm].item()] for elm in range(len(colnames))]))
    print(' '.join(['%2d' % permutation[elm].item() for elm in range(len(colnames))]))
    print('Cross validation log likelihood with %d split : %+.6f' % (n_split, mean_valid_loglik))
    print(edge_frequency.to_string())
    print('=' * 100)
    print('=' * 100)
    return valid_loglik_list, adj_mat_list, info_str_list


TOPORD_RUNTIME = {'sachs': 10, 'child': 30, 'insurance': 60}


class TOPORD(object):
    def __init__(self, name: str):
        self.name = 'TOPORD-%s' % name
        self.data = load_R_data(dataname=name)
        self.maximize = False
        self.n_cores = 30  # to accommodate 2 processes in parallel
        self.filename = __file__
        self.runtime = TOPORD_RUNTIME[name]

    def __call__(self, x):
        permutation = x.numpy().flatten()
        valid_loglik_list, adj_mat_list, info_str_list = cross_valiation_cat2cat_regression(
            permutation=permutation, data=self.data, valid_ratio=0.2, alpha=0.5, lamda=1e-2, threshold=0.1, n_init=5)
        neg_log_lik = np.mean(valid_loglik_list) * -1
        return torch.empty((1, 1), device=x.device).fill_(neg_log_lik)


def test_run(data):
    n_data, n_variables = data.shape
    n_train = int(n_data * (1 - 0.5))
    train_data = data.iloc[:n_train]
    valid_data = data.iloc[n_train:]
    permutation = np.random.RandomState(123).permutation(n_variables)
    regression_given_permutation(train_data=train_data, valid_data=valid_data, permutation=permutation, verbose=True)
    pass


if __name__ == '__main__':
    test_run(load_R_data(dataname='sachs'))
    exit(0)

    parser = argparse.ArgumentParser(description='Permutation Bayesian Optimization')
    parser.add_argument('--data', dest='data', type=str)
    parser.add_argument('--permutation', dest='permutation', type=str)

    args = parser.parse_args()
    start_time = time.time()
    evaluator = TOPORD(args.data)
    print('%12.6f seconds' % (time.time() - start_time))
    eval = evaluator(torch.tensor([int(elm) for elm in args.permutation.split(',')], dtype=torch.int))
    print('%12.6f seconds' % (time.time() - start_time))
    for _ in range(5):
        print(float(eval))
