from typing import Callable, Tuple, List, Dict, Optional

import os
import argparse
import time
from datetime import datetime
import pickle
import dill
from copy import deepcopy
from pathos import multiprocessing
import multiprocess.context as ctx

import torch
from torch import Tensor

import gpytorch
from gpytorch.kernels import ScaleKernel

from LAW2ORDER.gpytorch_bop.kernels import EnumPermutationKernel, NeuralKernelNetwork
from LAW2ORDER.gpytorch_bop.kernels.permutation_kernels import \
    HammingKernel, PositionKernel
from LAW2ORDER.surrogate.gp_models import ExactGPRegression, optimize_mll_parallel, pick_best_model
from LAW2ORDER.acquisition.function import expected_improvement, probability_improvement, \
    m_hat_estimate_permutation, optimization_as_estimation, gram_cholesky_lower_inv
from LAW2ORDER.acquisition.batch_acquisition import BaseDPPBatchAcquisition, BUCBBatchAcquisition, \
    OriginalDPPBatchAcquisition, WeightedDPPBatchAcquisition, MultiObjectiveEnsembleBatchAcquisition, \
    qPointBatchAcquisition
from LAW2ORDER.numerical_optimizer.utils import random_permutations, OPTIM_N_INITS
from LAW2ORDER.experiments.data import QAP, FSP, TSP
from LAW2ORDER.experiments.config_file_path import EXP_DIR, generate_bo_exp_dirname

ctx._force_start_method('spawn')
N_MODEL_INITS = 10
ACQ_OPTIM_N_INITS = 30


def suggest_batch(gp_model: ExactGPRegression, acquisition_type: str, batch_size: int,
                  data_x: Tensor, data_y: Tensor, maximize: bool = False) -> Tuple[Tensor, str]:
    data_y_mean, data_y_std = torch.mean(data_y).item(), torch.std(data_y).item()
    normalized_data_y = (data_y - data_y_mean) / data_y_std

    # set and optimize the surrogate model
    gp_model.set_train_data(inputs=data_x, targets=normalized_data_y, strict=False)
    model_dict = dict()
    mll_dict = dict()
    optimizer_dict = dict()
    param_init_dict = dict()
    for i in range(N_MODEL_INITS):
        model = deepcopy(gp_model)
        model_name = '%s(%02d)' % (model.kernel_name_str(), i)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood=model.likelihood, model=model)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model_dict[model_name] = model
        mll_dict[model_name] = mll
        optimizer_dict[model_name] = optimizer
        param_init_dict[model_name] = False if i == 0 else True

    negative_mll, optimized_model_dict, mll_dict, optimizer_dict = optimize_mll_parallel(
        train_x=data_x, train_y=normalized_data_y,
        model_dict=model_dict, mll_dict=mll_dict, optimizer_dict=optimizer_dict, param_init_dict=param_init_dict)
    best_model_name, info_str_surrogate = pick_best_model(negative_mll=negative_mll, model_dict=optimized_model_dict)
    best_model = optimized_model_dict[best_model_name]
    info_str_best = 'Best model %s with Likelihood Noise Variance : %.10f' \
                    % (best_model_name, best_model.likelihood.noise.data.item())

    # configure and optimize the acquisition function
    batch_type = acquisition_type.split('-')[0]
    if batch_type in ['DPP', 'WDPP']:
        use_eval_data = acquisition_type.split('-')[1] == 'posterior'
        acq_func_type = acquisition_type.split('-')[3]
        if acq_func_type == 'est':
            acq_function = optimization_as_estimation
            # Just to calculate mean_train and choleksy_lower_inv
            temp_acq = BaseDPPBatchAcquisition(
                gp_model=gp_model, data_x=data_x, data_y=data_y, batch_size=batch_size, acq_function=acq_function,
                use_eval_data=False, mcmc_step=None, maximize=maximize)
            incumbent = m_hat_estimate_permutation(
                gp_model=best_model, data_x=data_x, data_y=normalized_data_y, mean_train=temp_acq.mean_train,
                cholesky_lower_inv=temp_acq.cholesky_lower_inv, maximize=maximize)
            print('In EST, incumbent is %+.4E and best in data is %+.4E'
                  % (float(incumbent), (torch.max(normalized_data_y).item() if maximize
                                        else torch.min(normalized_data_y).item())))
            if maximize:
                assert incumbent >= torch.max(normalized_data_y)
            else:
                assert incumbent <= torch.min(normalized_data_y)
        elif acq_func_type == 'ei':
            acq_function = expected_improvement
            incumbent = (torch.max(normalized_data_y) if maximize else torch.min(normalized_data_y)).item()
        else:
            raise NotImplementedError

    if batch_type == 'DPP':
        mcmc_step = None if acquisition_type.split('-')[2] == 'MAX' else 20000
        acquisition = OriginalDPPBatchAcquisition(
            gp_model=best_model, data_x=data_x, data_y=normalized_data_y,
            batch_size=batch_size, acq_function=acq_function, mcmc_step=mcmc_step,
            maximize=maximize, incumbent=incumbent)
    elif batch_type == 'WDPP':
        assert acquisition_type.split('-')[2] in ['SAMPLE', 'MAX']
        mcmc_step = None if acquisition_type.split('-')[2] == 'MAX' else 20000
        acq_func_type = acquisition_type.split('-')[3]
        if acq_func_type == 'est':
            # est value is less than zero almost always, thus the behavior below zero matters
            # for the regret bound proof, boundedness is necessary
            # in this case, bounded in the interval [0.1, 1.0]
            def weight_function(x):
                return 0.01 + 0.99 / (1 + torch.exp(-x * 0.2))
        elif acq_func_type == 'ei':
            # ei can be shown that bounded between 0 and 1.25 x (2 x pi) ^ -0.5 = 0.4987
            # to make it have positive lower bound it is shifted by 0.01
            # positive std_modular is needed for the regret bound proof
            def weight_function(x):
                return 0.01 + (1 - 1 / (1 + x))
        else:
            raise NotImplementedError
        acquisition = WeightedDPPBatchAcquisition(
            gp_model=best_model, data_x=data_x, data_y=normalized_data_y, batch_size=batch_size,
            acq_function=acq_function, weight_function=weight_function, use_eval_data=use_eval_data,
            mcmc_step=mcmc_step, maximize=maximize, incumbent=incumbent)
    elif batch_type == 'qPOINT':
        acq_func_type = acquisition_type.split('-')[1]
        if acq_func_type == 'ei':
            acq_function = expected_improvement
            incumbent = (torch.max(normalized_data_y) if maximize else torch.min(normalized_data_y)).item()
        elif acq_func_type == 'pi':
            acq_function = probability_improvement
            incumbent = (torch.max(normalized_data_y) if maximize else torch.min(normalized_data_y)).item()
        elif acq_func_type == 'est':
            acq_function = optimization_as_estimation
            mean_train = best_model.mean_module(data_x).detach()
            cholesky_lower_inv = gram_cholesky_lower_inv(gp_model=best_model, data_x=data_x)
            incumbent = m_hat_estimate_permutation(
                gp_model=best_model, data_x=data_x, data_y=normalized_data_y, mean_train=mean_train,
                cholesky_lower_inv=cholesky_lower_inv, maximize=maximize)
        else:
            raise NotImplementedError
        acquisition = qPointBatchAcquisition(
            gp_model=best_model, data_x=data_x, data_y=normalized_data_y, batch_size=batch_size,
            acq_function=acq_function, maximize=maximize, incumbent=incumbent)
    elif batch_type == 'MACE':
        third_acq_type = acquisition_type.split('-')[1]
        acquisition = MultiObjectiveEnsembleBatchAcquisition(
            gp_model=gp_model, data_x=data_x, data_y=normalized_data_y, batch_size=batch_size,
            third_acq_type=third_acq_type, maximize=maximize)
    elif batch_type == 'BUCB':
        acquisition = BUCBBatchAcquisition(gp_model=gp_model, data_x=data_x, data_y=data_y,
                                           batch_size=batch_size, maximize=maximize)
    else:
        raise NotImplementedError

    next_x_batch, batch_acq_values, info_str_acq_optim = acquisition()

    print(info_str_surrogate)
    print(info_str_acq_optim)

    already_in = torch.any(torch.all(torch.eq(data_x.view(data_x.size()[0], 1, -1),
                                              next_x_batch.view(1, next_x_batch.size()[0], -1)), dim=-1))
    if already_in:
        print(dirname)
        raise AssertionError('Trying to evaluate an already evaluated point')

    for (name_dst, p_dst), (name_src, p_src) in zip(gp_model.named_parameters(), best_model.named_parameters()):
        assert name_dst == name_src
        p_dst.data = p_src.data
    # Now, the argument 'gp_model' has been updated to be the best_model, so that they can be used in following rounds

    return next_x_batch, '\n'.join([info_str_surrogate, info_str_best, info_str_acq_optim])


def evaluate_batch(evaluator: Callable, x_batch: Tensor) -> Tensor:
    n_eval_cores = evaluator.n_cores if hasattr(evaluator, 'n_cores') else 1
    pool = multiprocessing.Pool(max(1, min(multiprocessing.cpu_count() // n_eval_cores, x_batch.size()[0])))
    args_list = []
    for i in range(x_batch.size()[0]):
        args_list.append((x_batch[i], i))

    def eval_wrapper(x, batch_id):
        return batch_id, evaluator(x)

    ind, y = list(zip(*pool.starmap_async(eval_wrapper, args_list).get()))
    batch_y = torch.empty(x_batch.size()[0], 1, device=x_batch.device)
    batch_y[list(ind)] = torch.cat(y).view(-1, 1)
    return batch_y


def init_bo(kernel_type: str, acquisition_type: str,
            evaluator: Callable, batch_size: int, seed: Optional[int] = None, exp_dir: str = EXP_DIR):
    if seed is not None:
        assert seed < 10000
    p_size = evaluator.permutation_size
    kernel = ScaleKernel(PositionKernel(permutation_size=p_size))
    gp_model = ExactGPRegression(train_x=None, train_y=None, kernel=kernel)
    data_x = random_permutations(p_size=p_size, n_perm=OPTIM_N_INITS, seed=seed)
    data_y = evaluate_batch(evaluator=evaluator, x_batch=data_x)
    batch_idx = torch.zeros(OPTIM_N_INITS).long()
    bo_time = torch.zeros(OPTIM_N_INITS)
    eval_time = torch.zeros(OPTIM_N_INITS)
    time_stamp = [None for _ in range(OPTIM_N_INITS)]
    for i in range(OPTIM_N_INITS):
        eval_start_time = time.time()
        eval_time[i] = time.time() - eval_start_time
        time_stamp[i] = time.time()

    bo_data = {'gp_model': gp_model, 'acquisition_type': acquisition_type, 'batch_size': batch_size,
               'data_x': data_x, 'data_y': data_y,
               'evaluator': evaluator, 'batch_idx': batch_idx,
               'bo_time': bo_time, 'eval_time': eval_time, 'time_stamp': time_stamp}
    dir_name = generate_bo_exp_dirname(
        gp_model=bo_data['gp_model'], acquisition_type=acquisition_type,
        evaluator=evaluator, batch_size=batch_size, seed=seed, exp_dir=EXP_DIR, with_time_tag=True)
    with open(os.path.join(exp_dir, dir_name, '%s_%04d.pkl' % (dir_name, OPTIM_N_INITS)), 'wb') as f:
        dill.dump(bo_data, f)
    print_bo(bo_data, dir_name)
    return dir_name


def load_bo(dir_name: str, exp_dir: str) -> Dict:
    filename_list = [elm for elm in os.listdir(os.path.join(exp_dir, dir_name))
                     if os.path.isfile(os.path.join(exp_dir, dir_name, elm)) and elm[-4:] == '.pkl']
    last_filename_postfix = max([os.path.splitext(elm)[0].split('_')[-1] for elm in filename_list])
    last_filename = [elm for elm in filename_list
                     if ('%s.pkl' % last_filename_postfix) == elm[-(4 + len(last_filename_postfix)):]][0]
    with open(os.path.join(exp_dir, dir_name, last_filename), 'rb') as f:
        bo_data = pickle.load(f)
    return bo_data


def save_bo(bo_data: Dict, dir_name: str, exp_dir: str):
    with open(os.path.join(exp_dir, dir_name, '%s_%04d.pkl' % (dir_name, bo_data['data_y'].numel())), 'wb') as f:
        dill.dump(bo_data, f)


def run_bo(gp_model: ExactGPRegression, acquisition_type: str, evaluator: Callable, batch_size: int,
           data_x: torch.Tensor, data_y: torch.Tensor, batch_idx: torch.Tensor,
           bo_time: torch.Tensor, eval_time: torch.Tensor, time_stamp: List, **kwargs):
    assert data_x.size()[0] == bo_time.size()[0] == batch_idx.size()[0] \
           == data_y.size()[0] == eval_time.size()[0] == len(time_stamp)

    bo_start_time = time.time()
    next_x_batch, info_str_suggest = suggest_batch(
        gp_model=gp_model, acquisition_type=acquisition_type, batch_size=batch_size,
        data_x=data_x, data_y=data_y, maximize=evaluator.maximize)
    bo_time = torch.cat([bo_time, torch.ones(batch_size) * (time.time() - bo_start_time)], dim=0)
    data_x = torch.cat([data_x, next_x_batch.view(batch_size, -1).long()], dim=0)
    batch_idx = torch.cat([batch_idx, torch.ones(batch_size).long() * (batch_idx[-1] + 1)], dim=0)

    eval_start_time = time.time()
    next_y_batch = evaluate_batch(evaluator=evaluator, x_batch=next_x_batch)
    if data_y.ndim == 1:
        data_y = torch.cat([data_y, next_y_batch.view(-1)], dim=0)
    elif data_y.ndim == 2:
        data_y = torch.cat([data_y, next_y_batch.view(-1, 1)], dim=0)
    else:
        raise ValueError
    eval_time = torch.cat([eval_time, torch.ones(batch_size) * (time.time() - eval_start_time)], dim=0)
    time_stamp += [time.time() for _ in range(batch_size)]

    bo_data = {'gp_model': gp_model, 'acquisition_type': acquisition_type, 'batch_size': batch_size,
               'data_x': data_x, 'data_y': data_y,
               'evaluator': evaluator, 'batch_idx': batch_idx,
               'bo_time': bo_time, 'eval_time': eval_time, 'time_stamp': time_stamp}

    return bo_data, info_str_suggest


def print_bo(bo_data: Dict, dir_name: Optional[str] = None,
             print_prefix: Optional[str] = None, print_postfix: Optional[str] = None, exp_dir: str = EXP_DIR) -> None:
    maximize = bo_data['evaluator'].maximize
    torch_argopt = torch.argmax if maximize else torch.argmin
    torch_opt = torch.max if maximize else torch.min
    data_y = bo_data['data_y'].view(-1)
    batch_idx = bo_data['batch_idx'].view(-1)
    opt_in_batch_ind = torch.zeros_like(batch_idx).long()
    cumopt = torch.zeros_like(data_y)
    for j in range(batch_idx[-1].item() + 1):
        batch_member_idx = torch.nonzero(batch_idx == j).view(-1)
        prev_batch_member_idx = torch.nonzero(batch_idx <= j).view(-1)
        opt_member_idx = batch_member_idx[torch_argopt(data_y[batch_member_idx])]
        opt_in_batch_ind[batch_member_idx] = opt_member_idx
        cumopt[batch_member_idx] = torch_opt(data_y[prev_batch_member_idx]).item()
    bo_time = bo_data['bo_time'].view(-1)
    eval_time = bo_data['eval_time'].view(-1)
    time_stamp = bo_data['time_stamp']
    print_str_list = [] if print_prefix is None else [print_prefix]
    for i, (b_i, o_i_b_i, y, y_opt, t_bo, t_eval) \
            in enumerate(zip(batch_idx, opt_in_batch_ind, data_y, cumopt, bo_time, eval_time)):
        if (i == 0) or (batch_idx[i - 1] != batch_idx[i]):
            print_str_list.append(
                '%4d-th batch eval (Suggestion : %4d sec. / Evaluation : %4d sec.) %s:%+12.8f %s' %
                (b_i, t_bo, t_eval, 'MAX' if maximize else 'MIN', y_opt, '<----' if data_y[o_i_b_i] == y_opt else ''))
            cnt = 1
        print_str_list.append(
            '                  %3d : %+12.8f %s' % (cnt, y, ('OPT in %4d-th batch' % b_i) if i == o_i_b_i else ''))
        cnt += 1
    print_str_list.append('=' * 100)
    for i, (b_i, o_i_b_i, y, y_opt, t_bo, t_eval, t_stamp) \
            in enumerate(zip(batch_idx, opt_in_batch_ind, data_y, cumopt, bo_time, eval_time, time_stamp)):
        if (i == 0) or (batch_idx[i - 1] != batch_idx[i]):
            print_str_list.append(
                '%s %4d-th batch eval (Suggestion : %4d sec. / Evaluation : %4d sec.) %s:%+12.8f %s' %
                (datetime.fromtimestamp(t_stamp).strftime("%Y/%m/%d, %H:%M:%S"), b_i, t_bo, t_eval,
                 'MAX' if maximize else 'MIN', y_opt, '<----' if data_y[o_i_b_i] == y_opt else ''))
    if print_postfix is not None:
        print_str_list.append(print_postfix)
    print_str = '\n'.join(print_str_list)
    if dir_name is not None:
        if not os.path.exists(os.path.join(exp_dir, dir_name, 'log')):
            os.makedirs(os.path.join(exp_dir, dir_name, 'log'))
        with open(os.path.join(exp_dir, dir_name, 'log', '%s_%04d.log' % (dir_name, data_y.numel())), 'wt') as f:
            f.write(print_str)
    print(print_str)
    print('=' * 100)
    print('Experiment directory : %s' % dir_name)
    print('=' * 100)
    print('=' * 100)


def continue_bo(dir_name: str, rounds: int = 1, max_eval: Optional[int] = None):
    batch_size = load_bo(dir_name=dir_name, exp_dir=EXP_DIR)['batch_size']
    if max_eval is None:
        if batch_size == 5:
            max_eval = OPTIM_N_INITS + 5 * 100
        elif batch_size == 10:
            max_eval = OPTIM_N_INITS + 10 * 80
        elif batch_size == 20:
            max_eval = OPTIM_N_INITS + 20 * 40
        elif batch_size == 30:
            max_eval = OPTIM_N_INITS + 30 * 30
        else:
            raise ValueError('Specify the maximum number of evaluations')
    for i in range(rounds):
        bo_data = load_bo(dir_name=dir_name, exp_dir=EXP_DIR)
        if bo_data['data_y'].numel() >= max_eval:
            break
        bo_data, info_str_run_bo = run_bo(**bo_data)
        save_bo(bo_data=bo_data, dir_name=dir_name, exp_dir=EXP_DIR)
        print_bo(bo_data, dir_name, print_prefix=info_str_run_bo)
    return dir_name


# It can be shown that EI is bounded above provided that the kernel is bounded above
# WDPP-prior-MAX resembles LP-UCB, thus included
# WDPP-posterior-MAX and WDPP-posterior-SAMPLE are kind of improvement upon DPP-posterior thus included
ACQUISITION_TYPE_LIST = [
    'DPP-posterior-MAX-est', 'DPP-posterior-SAMPLE-est', 'MACE-ucb', 'MACE-est', 'BUCB',
    'WDPP-posterior-MAX-est', 'WDPP-posterior-MAX-ei', 'WDPP-prior-MAX-est', 'WDPP-prior-MAX-ei',
    'qPOINT-ei', 'qPOINT-est'
    # The proof of SAMPLE method in Batched Gaussian Process Bandit Optimization via Determinantal Point Processes
    # is wrong, because it relies on the assumption that the sampled points are maximum of the considered objectives
    # 'WDPP-prior-SAMPLE-est-shift', 'WDPP-prior-SAMPLE-ei-shift',
    # 'WDPP-posterior-SAMPLE-est-shift', 'WDPP-posterior-SAMPLE-ei-shift'
]


# There are more benchmarks, but we choose below considering diversity in problem types and permutation sizes
BENCHMARK_LIST = [
    'QAP-chr12a',   # 12
    'QAP-nug22',    # 22
    'QAP-esc32a',   # 32
    'QAP-sko42',    # 42
    'FSP-car5',     # 10
    'FSP-hel2',     # 20
    'FSP-reC19',    # 30
    'FSP-reC31',    # 50
    'TSP-burma14',  # 14
    'TSP-bayg29',   # 29
    'TSP-att48',    # 48
    'TSP-st70'      # 70
]
BENCHMARK_OPTIMUM = {
    'QAP-chr12a': 9552,
    'QAP-nug22': 3596,
    'QAP-esc32a': None,  # 130,
    'QAP-sko42': None,   # this is bound, feasible solution is 7205962
    'FSP-car5': None,
    'FSP-hel2': None,
    'FSP-reC19': None,
    'FSP-reC31': None,
    'TSP-burma14': None,
    'TSP-bayg29': None,
    'TSP-att48': None,
    'TSP-st70': None
}
BENCHMARK_SIZE = {
    'QAP-chr12a': 12,
    'QAP-nug22': 22,
    'QAP-esc32a': 32,
    'QAP-sko42': 42,
    'FSP-car5': 10,
    'FSP-hel2': 20,
    'FSP-reC19': 30,
    'FSP-reC31': 50,
    'TSP-burma14': 14,
    'TSP-bayg29': 29,
    'TSP-att48': 48,
    'TSP-st70': 70
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Permutation Bayesian Optimization')
    parser.add_argument('--objective', dest='objective', type=str)
    parser.add_argument('--seed', dest='seed', type=int, default=None)
    parser.add_argument('--batch', dest='batch', type=int)
    parser.add_argument('--kernel', dest='kernel', type=str, help='position')
    parser.add_argument('--acquisition', dest='acquisition', type=str,
                        help=' '.join(ACQUISITION_TYPE_LIST))
    parser.add_argument('--dirname', dest='dirname', type=str, default=None)
    parser.add_argument('--rounds', dest='rounds', type=int, default=1)
    parser.add_argument('--max_eval', dest='max_eval', type=int, default=None)

    args = parser.parse_args()
    # Thompson sampling is infeasible
    # While, in continuous search space cases,
    # GP posterior sample can be approximated by random features whose analytic form is known
    # in discrete cases, random features are not given in closed form in general.
    # Therefore, sampling cost m^3 where m is the number of points tested for GP posterior sample
    # Predictive Entropy Search for Efficient GlobalOptimization of Black-box Functions; Jose Miguel Hernandez-Lobato,
    # Matthew W. Hoffman, Zoubin Ghahramani
    assert (args.objective is None) == (args.seed is None) == (args.kernel is None) == (args.acquisition is None)
    assert (args.objective is None) != (args.dirname is None)
    if args.objective is not None:
        assert isinstance(args.kernel, str)
        assert isinstance(args.acquisition, str)
        assert args.acquisition in ACQUISITION_TYPE_LIST
        evaluator_type, evaluator_name = args.objective.split('-')
        assert evaluator_type.upper() in ['QAP', 'FSP', 'TSP']
        dirname = init_bo(kernel_type=args.kernel, acquisition_type=args.acquisition,
                          evaluator=globals()[evaluator_type](name=evaluator_name),
                          batch_size=args.batch, seed=args.seed)
    else:
        dirname = args.dirname
    continue_bo(dir_name=dirname, rounds=args.rounds, max_eval=args.max_eval)
