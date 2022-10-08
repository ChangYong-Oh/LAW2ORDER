from typing import Callable, Tuple, Optional, List

import time
from tqdm import tqdm

from pathos import multiprocessing
import multiprocess.context as ctx

import numpy as np

import torch
from torch import Tensor
from torch.optim import Adam

from LAW2ORDER.torch_bop.distributions import PlackettLuce
from LAW2ORDER.gradient_estimators import backward_reinforce_with_replacement
# from LAW2ORDER.numerical_optimizer.simulated_annealing import PermutationAnnealer
from LAW2ORDER.numerical_optimizer.hill_climbing import \
    DiscreteHillClimbing, PermutationHillClimbing, PermutationPairHillClimbing
from LAW2ORDER.acquisition.function import random_discrete

ctx._force_start_method('spawn')
# HILL_CLIMBING_N_CORES_USED = 16  # permutation
HILL_CLIMBING_N_CORES_USED = 8  # discrete


def optimization_init_permutation_points(objective: Callable, data_x: Tensor, batch_size: int, maximize: bool = True
                                         ) -> Tuple[Tensor, Tensor, str]:
    n_data, p_size = data_x.size()
    n_rand = 20000
    rand_points = torch.zeros(n_rand, p_size).long()
    for i in range(n_rand):
        rand_points[i] = torch.randperm(p_size)
    obj_values = objective(torch.cat([data_x, rand_points], dim=0))
    obj_value_data = obj_values[:n_data]
    obj_value_rand = obj_values[n_data:]

    n_init_random1 = 10  # topk points among random
    n_init_random2 = 10  # purely random
    obj_value_rand_topk, obj_value_rand_topkind = torch.topk(obj_value_rand, k=n_init_random1, largest=maximize)
    rand_inds = list(obj_value_rand_topkind.numpy()) + [elm.item() for elm
                                                        in torch.randperm(n_rand)[:n_init_random1 + n_init_random2]
                                                        if elm not in obj_value_rand_topkind][:n_init_random2]
    init_from_rand = rand_points[rand_inds]

    n_init_sub1 = 10  # topk points among evaluated data
    n_init_sub2 = 10  # random points among evaluated data
    obj_value_data_ordered, obj_value_data_orderedind = torch.sort(obj_value_data, descending=maximize, dim=0)
    # points_to_avoid = data_x[range(max(0, n_data - 2 * batch_size), n_data)]
    points_to_avoid = data_x

    if n_data > 2 * batch_size:
        n_recent = 2 * batch_size
    elif n_data > batch_size:
        n_recent = batch_size
    else:
        n_recent = 0
    init_points_ind1 = obj_value_data_orderedind[obj_value_data_orderedind < n_data - n_recent][:n_init_sub1]
    # Even though in above line, recent points are excluded,
    # but random choice below stochastically allows inclusion of some of them
    init_points_ind2 = torch.LongTensor(np.random.choice(list((set(range(n_data)).difference(
        set(init_points_ind1.numpy())))), size=n_init_sub2, replace=False))
    init_points_ind = torch.cat([init_points_ind1, init_points_ind2])
    init_from_data = data_x[init_points_ind]

    init_points = torch.cat([init_from_data, init_from_rand])

    info_str_list = ['top%d acq values on evaluated data' % n_init_sub1,
                     '    ' + '  '.join(['%.10f' % elm for elm in obj_value_data_ordered[:n_init_sub1]])]
    return init_points, points_to_avoid, '\n'.join(info_str_list)


def permutation_hill_climbing(
        objective: Callable, init_points: Tensor, points_to_avoid: Optional[Tensor] = None, maximize: bool = True,
        constraint: Optional[Callable] = None) -> Tuple[Tensor, float, int]:
    start_time = time.time()
    n_processes = max(1, multiprocessing.cpu_count() // HILL_CLIMBING_N_CORES_USED)
    n_init = init_points.size()[0]
    pool = multiprocessing.Pool(n_processes)

    def optimize_wrapper(idx):
        climber = PermutationHillClimbing(objective=objective, initial_state=init_points[idx],
                                          states_to_avoid=points_to_avoid, minimize=not maximize, constraint=constraint)
        climb_input, climb_output = climber.climb()
        return idx, climb_input, climb_output, climber.cnt_update

    args_list = [(elm,) for elm in range(n_init)]
    ind, climb_input_list, climb_output_list, cnt_list = \
        list(zip(*pool.starmap_async(optimize_wrapper, args_list).get()))

    climb_inputs = torch.zeros_like(init_points)
    climb_outputs = torch.zeros(n_init)
    climb_cnt = torch.zeros(n_init).long()
    for i in range(n_init):
        climb_inputs[ind[i]] = climb_input_list[i]
        climb_outputs[ind[i]] = climb_output_list[i]
        climb_cnt[ind[i]] = cnt_list[i]

    best_ind = np.nanargmax(climb_outputs.numpy()) if maximize else np.nanargmin(climb_outputs.numpy())
    best_input, best_output = climb_inputs[best_ind], climb_outputs[best_ind].item()

    print('     %12d seconds to optimize with %d initializations with %d processes' %
          (time.time() - start_time, len(args_list), n_processes))

    return best_input, best_output, best_ind


def optimization_init_discrete_points(
        n_cat_list: List[int], objective: Callable, data_x: Tensor, batch_size: int, maximize: bool = True
) -> Tuple[Tensor, Tensor, str]:
    n_data, n_dim = data_x.size()
    n_rand = 20000
    rand_points = random_discrete(n_cat_list=n_cat_list, n=n_rand, device=data_x.device)
    obj_values = objective(torch.cat([data_x, rand_points], dim=0))
    assert obj_values.dim() == 1
    obj_value_data = obj_values[:n_data]
    obj_value_rand = obj_values[n_data:]

    n_init_random1 = 10  # topk points among random
    n_init_random2 = 10  # purely random
    obj_value_rand_topk, obj_value_rand_topkind = torch.topk(obj_value_rand, k=n_init_random1, largest=maximize)
    rand_inds = list(obj_value_rand_topkind.numpy()) + [elm.item() for elm
                                                        in torch.randperm(n_rand)[:n_init_random1 + n_init_random2]
                                                        if elm not in obj_value_rand_topkind][:n_init_random2]
    init_from_rand = rand_points[rand_inds]

    n_init_sub1 = 10  # topk points among evaluated data
    n_init_sub2 = 10  # random points among evaluated data
    obj_value_data_ordered, obj_value_data_orderedind = torch.sort(obj_value_data, descending=maximize, dim=0)
    # points_to_avoid = data_x[range(max(0, n_data - 2 * batch_size), n_data)]
    points_to_avoid = data_x

    if n_data > 2 * batch_size:
        n_recent = 2 * batch_size
    elif n_data > batch_size:
        n_recent = batch_size
    else:
        n_recent = 0
    init_points_ind1 = obj_value_data_orderedind[obj_value_data_orderedind < n_data - n_recent][:n_init_sub1]
    # Even though in above line, recent points are excluded,
    # but random choice below stochastically allows inclusion of some of them
    init_points_ind2 = torch.LongTensor(np.random.choice(list((set(range(n_data)).difference(
        set(init_points_ind1.numpy())))), size=n_init_sub2, replace=False))
    init_points_ind = torch.cat([init_points_ind1, init_points_ind2])
    init_from_data = data_x[init_points_ind]

    init_points = torch.cat([init_from_data, init_from_rand])

    info_str_list = ['top%d acq values on evaluated data' % n_init_sub1,
                     '    ' + '  '.join(['%.10f' % elm for elm in obj_value_data_ordered[:n_init_sub1]])]
    return init_points, points_to_avoid, '\n'.join(info_str_list)


def discrete_hill_climbing(
        adj_mat_list: List[Tensor], objective: Callable, init_points: Tensor, points_to_avoid: Optional[Tensor] = None,
        maximize: bool = True, constraint: Optional[Callable] = None) -> Tuple[Tensor, float, int]:
    start_time = time.time()
    n_processes = max(1, multiprocessing.cpu_count() // HILL_CLIMBING_N_CORES_USED)
    n_init = init_points.size()[0]
    pool = multiprocessing.Pool(n_processes)

    def optimize_wrapper(idx):
        climber = DiscreteHillClimbing(adj_mat_list=adj_mat_list, objective=objective, initial_state=init_points[idx],
                                       states_to_avoid=points_to_avoid, minimize=not maximize, constraint=constraint)
        climb_input, climb_output = climber.climb()
        return idx, climb_input, climb_output, climber.cnt_update

    args_list = [(elm,) for elm in range(n_init)]
    ind, climb_input_list, climb_output_list, cnt_list = \
        list(zip(*pool.starmap_async(optimize_wrapper, args_list).get()))

    climb_inputs = torch.zeros_like(init_points)
    climb_outputs = torch.zeros(n_init)
    climb_cnt = torch.zeros(n_init).long()
    for i in range(n_init):
        climb_inputs[ind[i]] = climb_input_list[i]
        climb_outputs[ind[i]] = climb_output_list[i]
        climb_cnt[ind[i]] = cnt_list[i]

    best_ind = np.nanargmax(climb_outputs.numpy()) if maximize else np.nanargmin(climb_outputs.numpy())
    best_input, best_output = climb_inputs[best_ind], climb_outputs[best_ind].item()

    print('     %12d seconds to optimize with %d initializations with %d processes' %
          (time.time() - start_time, len(args_list), n_processes))

    return best_input, best_output, best_ind


def permutation_pair_hill_climbing(
        objective: Callable, init_points: Tensor, points_to_avoid: Optional[Tensor] = None, maximize: bool = True,
        constraint: Optional[Callable] = None) -> Tuple[Tensor, float, int]:
    """
    """
    climb_inputs = init_points.new_zeros(init_points.size()).long()
    climb_outputs = init_points.new_empty(init_points.size()[0]).float()
    climb_cnt = []
    for i in tqdm(range(climb_inputs.size(0))):
        climber = PermutationPairHillClimbing(
            objective=objective, initial_state=init_points[i],
            states_to_avoid=points_to_avoid, minimize=not maximize, constraint=constraint)
        climb_inputs[i], climb_outputs[i] = climber.climb()
        climb_cnt.append(climber.cnt_update)
    best_ind = (torch.argmax(climb_outputs) if maximize else torch.argmin(climb_outputs)).item()
    best_input, best_output = climb_inputs[best_ind], climb_outputs[best_ind].item()
    return best_input, best_output, best_ind


def optimize_acquisition_on_permutation(
        acq_function: Callable, permutation_size: int, init_points: Tensor, method: str,
        seconds: Optional[float] = None, points_to_avoid: Optional[Tensor] = None):
    """
    Important and differentiating characteristics of acquisition function optimization
    1. The landscape of acquisition functions tends to be flat in most of regions with a low acquisition value.
    2. The acquisition function landscape changes quite marginally if the number of new acquisitions is small.
    3. Such characteristics are more prominent when a search space is combinatorial, which may explain why local search
       can outperforms others.

    In preliminary experiments, it is shown that local search (Hill Climbing) outperforms SA in all tested cases of
    the acquisition function optimization.
    In Variational optimization experiments, it is shown that SA outperforms variational optimization.
    Variational optimization with low-variance gradient estimator tends to be a parameter making distribution is highly
    concentrated.
    Among many gradient estimators, the quality of optimized values is Unordered Set > VIMCO variant > PL-relax.
    Considering computational efficiency,  VIMCO variant > PL-relax >> Unordered Set.
    Overall, VIMCO variant is well-balanced between efficiency and performance.

    :param acq_function:
    :param permutation_size:
    :param init_points:
    :param method:
    :param seconds:
    :param points_to_avoid:
    :return:
    """
    start_time = time.time()
    if method.lower() == 'hc':  # hill climbing local
        climb_inputs = init_points.new_zeros(init_points.size()).long()
        climb_outputs = init_points.new_empty(init_points.size()[0]).float()
        climb_cnt = []
        for i in tqdm(range(climb_inputs.size()[0])):
            climber = PermutationHillClimbing(objective=acq_function, initial_state=init_points[i],
                                              states_to_avoid=points_to_avoid, minimize=False)
            climb_inputs[i], climb_outputs[i] = climber.climb()
            climb_cnt.append(climber.cnt_update)
        best_ind = torch.argmax(climb_outputs).item()
        best_input, best_output = climb_inputs[best_ind], climb_outputs[best_ind].item()
        info_str_list = ['Hill Climbing (%d seconds)' % (int(time.time() - start_time)),
                         'Num. of updates : ' + ' / '.join(['%d:%d' % (j, elm) for j, elm in enumerate(climb_cnt)]),
                         'FROM [%s] %.10f(%2d-th)' % (', '.join(['%d' % elm for elm in init_points[best_ind].numpy()]),
                                                      acq_function(init_points[best_ind].view(1, -1)).item(), best_ind),
                         '  TO [%s] %.10f' % (', '.join(['%d' % elm for elm in best_input.numpy()]), best_output)]
    elif method.lower() == 'sa':
        assert init_points.ndim == 2
        n_init_points = init_points.size()[0]
        annealer = PermutationAnnealer(objective=acq_function, permutation_size=permutation_size,
                                       states_to_avoid=points_to_avoid, minimize=False)
        annealer.updates = 0
        auto_schedule = annealer.auto(minutes=seconds / n_init_points / 60.0, steps=500)
        annealer.set_schedule(auto_schedule)
        annealed_inputs, annealed_outputs = [None for _ in range(n_init_points)], [None for _ in range(n_init_points)]
        for i in tqdm(range(n_init_points)):
            annealer.state = annealer.restricted_move(state=list(init_points[i].numpy()),
                                                      states_to_avoid=points_to_avoid)
            annealed_inputs[i], annealed_outputs[i] = annealer.anneal()
        best_ind = np.argmax(annealed_outputs).item()
        best_input = torch.LongTensor(annealed_inputs[best_ind])
        best_output = float(annealed_outputs[best_ind])
        info_str_list = ['Simulated annealing (%d seconds)' % (int(time.time() - start_time)),
                         'FROM [%s] %.10f' % (', '.join(['%d' % elm for elm in init_points[best_ind].numpy()]),
                                              acq_function(init_points[best_ind].view(1, -1)).item()),
                         '  TO [%s] %.10f' % (', '.join(['%d' % elm for elm in best_input.numpy()]), best_output)]
    elif method.lower() == 'vimco':
        k = 4
        initial_log_scores = torch.arange(init_points.size()[0], -1, -1).float()
        initial_log_scores = (initial_log_scores - torch.mean(initial_log_scores)) / torch.std(initial_log_scores) * 1.0
        initial_log_scores = initial_log_scores[torch.argsort(init_points)]
        log_scores = initial_log_scores.clone().detach_().requires_grad_(True)

        optimizer = Adam(params=[log_scores], lr=1.0)

        best_output = -np.inf
        best_input = None

        n_steps_for_runtime_estimate = 100
        for _ in range(n_steps_for_runtime_estimate):
            optimizer.zero_grad()
            distribution = PlackettLuce(log_scores=log_scores, recursive=False)
            output_samples, input_samples = backward_reinforce_with_replacement(
                integrand=acq_function, distribution=distribution, k=k, minimize=False)
            optimizer.step()
            best_sample_ind = torch.argmax(output_samples)
            if best_output < output_samples[best_sample_ind]:
                best_output = output_samples[best_sample_ind].item()
                best_input = input_samples[best_sample_ind]
        n_steps = int(seconds / ((time.time() - start_time) / n_steps_for_runtime_estimate))
        for _ in range(1, n_steps + 1):
            optimizer.zero_grad()
            distribution = PlackettLuce(log_scores=log_scores, recursive=False)
            output_samples, input_sample = backward_reinforce_with_replacement(
                integrand=acq_function, distribution=distribution, k=k, minimize=False)
            optimizer.step()
            best_sample_ind = torch.argmax(output_samples)
            if best_output < output_samples[best_sample_ind]:
                best_output = output_samples[best_sample_ind].item()
                best_input = input_samples[best_sample_ind]
        info_str_list = ['VIMCO variant (%d seconds)' % (int(time.time() - start_time)),
                         'FROM [%s] %.10f' % (', '.join(['%d' % elm for elm in init_points.numpy()]),
                                              acq_function(init_points.view(1, -1)).item()),
                         '  TO [%s] %.10f' % (', '.join(['%d' % elm for elm in best_input.numpy()]), best_output)]
    else:
        raise NotImplementedError
    info_str = '\n'.join(info_str_list)
    return best_input, best_output, info_str
