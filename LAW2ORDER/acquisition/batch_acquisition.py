from typing import Callable, Optional, Tuple, List

import math
import time

import numpy as np

import torch
from torch import Tensor

from gpytorch.utils.cholesky import psd_safe_cholesky

from LAW2ORDER.surrogate.gp_models import ExactGPRegression
from LAW2ORDER.acquisition.function import pred_mean_std, gram_cholesky_lower_inv, \
    expected_improvement, probability_improvement, confidence_bound, ucb_beta, \
    random_discrete, m_hat_estimate_discrete, m_hat_estimate_permutation, optimization_as_estimation
from LAW2ORDER.acquisition.optimization import \
    optimization_init_permutation_points, permutation_hill_climbing, \
    optimization_init_discrete_points, discrete_hill_climbing

from pymoo.algorithms.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.factory import get_selection, get_crossover, get_mutation
from pymoo.model.problem import Problem
from pymoo.util.termination.default import MultiObjectiveDefaultTermination


def hill_climbing(search_space_type: str, objective: Callable, init_data_x: Tensor, batch_size: int,
                  constraint: Optional[Callable] = None, adj_mat_list: Optional[List[Tensor]] = None):
    if search_space_type == 'discrete':
        n_cat_list = [elm.size()[0] for elm in adj_mat_list]
        init_points, points_to_avoid, _ = optimization_init_discrete_points(
            n_cat_list=n_cat_list, objective=objective, data_x=init_data_x, batch_size=batch_size, maximize=True)
        best_input, best_output, best_ind = discrete_hill_climbing(
            adj_mat_list=adj_mat_list, objective=objective, init_points=init_points, points_to_avoid=points_to_avoid,
            constraint=constraint, maximize=True)
    elif search_space_type == 'permutation':
        init_points, points_to_avoid, _ = optimization_init_permutation_points(
            objective=objective, data_x=init_data_x, batch_size=batch_size, maximize=True)
        best_input, best_output, best_ind = permutation_hill_climbing(
            objective=objective, init_points=init_points, points_to_avoid=points_to_avoid,
            constraint=constraint, maximize=True)
    else:
        raise NotImplementedError
    return best_input, best_output, best_ind, init_points, points_to_avoid


def optimization_summary(acquisition_func, points_in_batch: Tensor, init_points: Tensor, init_point_obj_values: Tensor,
        best_input_obj_value: float, best_input_acq_value: Tensor, best_ind: int) -> str:
    init_point_acq_values = acquisition_func(init_points).view(-1)
    n_inits = init_points.size()[0]
    idx_str_list = ['inits'] + ['%9d' % i for i in range(1, n_inits + 1)]
    obj_value_str_list = ['obj :'] + ['%+.2E' % init_point_obj_values[i].item() for i in range(n_inits)]
    acq_value_str_list = ['acq :'] + ['%+.2E' % init_point_acq_values[i].item() for i in range(n_inits)]
    optim_info_str = 'The result is obtained from %2d-th init where objective   : %+.4E -> %+.4E\n' \
                     '                                             acquisition : %+.4E -> %+.4E' % \
                     (best_ind + 1, init_point_obj_values[best_ind].item(), best_input_obj_value,
                      init_point_acq_values[best_ind].item(), best_input_acq_value.item())

    greedy_optim_info_str_list = [
        '%2d-th point in the batch' % (1 if points_in_batch is None else points_in_batch.size()[0] + 1),
        '  '.join(idx_str_list), '  '.join(obj_value_str_list), '  '.join(acq_value_str_list), optim_info_str]

    return '\n'.join(greedy_optim_info_str_list)


class BatchAcquisition(object):
    def __init__(self, gp_model: ExactGPRegression, data_x: Tensor, data_y: Tensor, batch_size: int,
                 maximize: bool = False, mcmc_step: Optional[int] = None, use_relevant_region: bool = False, **kwargs):
        """
        In the original BoTorch implementation, acquisition function performs unnecessary repetition of gram matrix
        computation, in this class, such unnecessary repeated computation is replaced with cached values,
        e.g. cholesky_lower_inv, mean_train
        This speeds up acquisition function optimization significantly
        :param gp_model:
        :param data_x:
        :param data_y:
        :param batch_size: size of batch
        :param maximize: whether the objective is maximized
        """
        self.gp_model = gp_model
        if hasattr(gp_model.covar_module.base_kernel, 'fourier_freq_list'):
            self.search_space_type = 'discrete'
            self.n_cat_list = [elm.size()[0] for elm in gp_model.covar_module.base_kernel.fourier_basis_list]
            self.adj_mat_list = gp_model.covar_module.base_kernel.adj_mat_list
            self.n_data, self.n_cats = data_x.size()
            self.p_size = None
            self.space_size = np.prod(self.n_cat_list)
        elif hasattr(gp_model.covar_module.base_kernel, 'permutation_size') or \
                hasattr(gp_model.covar_module.base_kernel, '_permutation_size'):
            self.search_space_type = 'permutation'
            self.n_data, self.p_size = data_x.size()
            self.n_cats = None
            self.space_size = math.factorial(self.p_size)
        else:
            raise NotImplementedError
        self.device = data_x.device
        self.cholesky_lower_inv = gram_cholesky_lower_inv(gp_model=gp_model, data_x=data_x)
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        self.mean_train = self.gp_model.mean_module(self.data_x).detach()
        self.maximize = maximize
        self.acq_function = None
        self.acq_kwargs = kwargs
        self.mcmc_step = mcmc_step
        if use_relevant_region:
            if self.search_space_type == 'discrete':
                beta_t = 2 * (np.sum(np.log(self.n_cat_list)) + np.log((self.n_data * np.pi) ** 2 / (6 * 0.05)))
            elif self.search_space_type == 'permutation':
                beta_t = 2 * (np.sum(np.log(np.arange(1, self.p_size + 1)))
                              + np.log((self.n_data * np.pi) ** 2 / (6 * 0.05)))
            else:
                raise NotImplementedError

            def relevant_region_objective(x):
                pred_mean, pred_std = pred_mean_std(
                    x=x, gp_model=self.gp_model, data_x=self.data_x, data_y=self.data_y,
                    mean_train=self.mean_train, cholesky_lower_inv=self.cholesky_lower_inv)
                return (pred_mean - beta_t ** 0.5 * pred_std) \
                    if self.maximize else -(pred_mean + beta_t ** 0.5 * pred_std)

            if self.search_space_type == 'discrete':
                init_points, _, _ = optimization_init_discrete_points(
                    n_cat_list=self.n_cat_list, objective=relevant_region_objective,
                    data_x=self.data_x, batch_size=self.batch_size, maximize=True)
                _, relevant_region_bound, _ = discrete_hill_climbing(
                    adj_mat_list=self.adj_mat_list, objective=relevant_region_objective,
                    init_points=init_points, points_to_avoid=None, maximize=True)
            elif self.search_space_type == 'permutation':
                init_points, _, _ = optimization_init_permutation_points(
                    objective=relevant_region_objective, data_x=self.data_x, batch_size=self.batch_size, maximize=True)
                _, relevant_region_bound, _ = permutation_hill_climbing(
                    objective=relevant_region_objective, init_points=init_points, points_to_avoid=None, maximize=True)
            else:
                raise NotImplementedError
            if not self.maximize:
                relevant_region_bound *= -1

            def in_relevant_region(x):
                pred_mean, pred_std = pred_mean_std(
                    x=x, gp_model=self.gp_model, data_x=self.data_x, data_y=self.data_y,
                    mean_train=self.mean_train, cholesky_lower_inv=self.cholesky_lower_inv)
                return pred_mean + 2 * beta_t ** 0.5 * pred_std > relevant_region_bound if self.maximize \
                    else pred_mean - 2 * beta_t ** 0.5 * pred_std < relevant_region_bound

            self.in_relevant_region = in_relevant_region
        else:
            self.in_relevant_region = None
        self.name = None

    def acquisition_value(self, x: Tensor) -> Tensor:
        # backprogpagation is not called via this class, so using detach() is fine
        pred_mean, pred_std = pred_mean_std(
            x=x, gp_model=self.gp_model, data_x=self.data_x, data_y=self.data_y,
            mean_train=self.mean_train, cholesky_lower_inv=self.cholesky_lower_inv)
        return self.acq_function(mean=pred_mean, sigma=pred_std, **self.acq_kwargs)

    def _kernel_computation(
            self, x1: Tensor, x2: Optional[Tensor] = None, diag: bool = False, use_eval_data: bool = True) -> Tensor:
        device = x1.device
        if diag:
            assert x2 is None
        # The case that x2 is None is handled automatically in covar_module.__call__
        gram_x1_x2 = self.gp_model.covar_module(x1, x2, diag=diag)
        gram_x1_x2 = (gram_x1_x2.evaluate() if not isinstance(gram_x1_x2, Tensor) else gram_x1_x2).detach()
        if not diag and x2 is None:
            gram_x1_x2 = gram_x1_x2 + torch.eye(x1.size()[0], device=device) * self.gp_model.likelihood.noise.detach()
        # the same form as the posterior conditioned on all evaluations
        if use_eval_data:
            gram_data_x1 = self.gp_model.covar_module(self.data_x, x1)
            gram_data_x1 = (gram_data_x1.evaluate() if not isinstance(gram_data_x1, Tensor) else gram_data_x1).detach()
            chol_lower_inv_gram_data_x1 = torch.mm(self.cholesky_lower_inv, gram_data_x1)
            if x2 is None:
                chol_lower_inv_gram_data_x2 = chol_lower_inv_gram_data_x1
            else:
                gram_data_x2 = self.gp_model.covar_module(self.data_x, x2)
                gram_data_x2 = (gram_data_x2.evaluate() if not isinstance(gram_data_x2, Tensor)
                                else gram_data_x2).detach()
                chol_lower_inv_gram_data_x2 = torch.mm(self.cholesky_lower_inv, gram_data_x2)
            if not diag:
                gram_x1_x2 = gram_x1_x2 - torch.mm(chol_lower_inv_gram_data_x1.t(), chol_lower_inv_gram_data_x2)
            else:
                gram_x1_x2 = gram_x1_x2 - torch.sum(chol_lower_inv_gram_data_x1 ** 2, dim=0)
        return gram_x1_x2

    def _optimization_summary(self, points_in_batch: Tensor, init_points: Tensor, init_point_obj_values: Tensor,
                              best_input_obj_value: float, best_input_acq_value: Tensor, best_ind: int) -> str:
        return optimization_summary(self.acquisition_value, points_in_batch, init_points, init_point_obj_values,
                                    best_input_obj_value, best_input_acq_value, best_ind)

    def greedy_batch_objective(self, points_in_batch: Optional[Tensor] = None) -> Callable:
        raise NotImplementedError

    def greedy_optimization(self, points_in_batch: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, str]:
        hill_climbing_objective = self.greedy_batch_objective(points_in_batch=points_in_batch)

        best_input, best_output, best_ind, init_points, points_to_avoid = hill_climbing(
            search_space_type=self.search_space_type, objective=hill_climbing_objective,
            init_data_x=self.data_x if points_in_batch is None else torch.cat([self.data_x, points_in_batch], dim=0),
            batch_size=self.batch_size, constraint=self.in_relevant_region,
            adj_mat_list=self.adj_mat_list if self.search_space_type == 'discrete' else None)

        best_input_acq_value = self.acquisition_value(best_input.view(1, -1))
        greedy_optim_info_str = self._optimization_summary(
            points_in_batch=points_in_batch, init_points=init_points,
            init_point_obj_values=hill_climbing_objective(init_points).view(-1),
            best_input_obj_value=best_output, best_input_acq_value=best_input_acq_value, best_ind=best_ind)

        return best_input.view(1, -1), best_input_acq_value, greedy_optim_info_str

    def _dpp_kernel(self, x1: Tensor, x2: Optional[Tensor] = None, diag: bool = False) -> Tensor:
        raise NotImplementedError

    def _dpp_mcmc(self, batch_points):
        """
        In Monte Carlo Markov Chain Algorithms for SamplingStrongly Rayleigh Distributions and Determinantal Point
        Processes; Nima Anari, Shayan Oveis Gharan, Alireza Rezaei, a guideline for the number of mcmc_step
        with an approximation guarantee is given
        however, it is proportional to the size of the set where k points are sampled, which is infeasible
        for permutation cases, so we resort to some computationally affordable number.
        Moreover, it start from a highly likely point using a greedy algorithm, it may be the case that the small number
        of steps is enough in practice.
        :param batch_points:
        :return:
        """
        device = batch_points.device
        curr_density = torch.det(self._dpp_kernel(batch_points))
        print('  MCMC begins to make %d transitions' % self.mcmc_step)
        mcmc_start_time = time.time()
        for _ in range(self.mcmc_step):
            i = torch.randint(self.batch_size, (1,)).item()
            if self.search_space_type == 'discrete':
                new_point = random_discrete(n_cat_list=self.n_cat_list, n=1, device=device).view(1, -1)
            elif self.search_space_type == 'permutation':
                new_point = torch.randperm(self.p_size, device=device).view(1, -1)
            else:
                raise NotImplementedError
            # random permutation is not contained in curr_batch and does satisfy constraint
            in_relevant_region = self.in_relevant_region(new_point) if hasattr(self, 'in_relevant_region') else True
            while torch.any(torch.all(torch.eq(batch_points, new_point), dim=1)) and in_relevant_region:
                if self.search_space_type == 'discrete':
                    new_point = random_discrete(n_cat_list=self.n_cat_list, n=1, device=device).view(1, -1)
                elif self.search_space_type == 'permutation':
                    new_point = torch.randperm(self.p_size, device=device).view(1, -1)
                else:
                    raise NotImplementedError
                in_relevant_region = self.in_relevant_region(new_point) if hasattr(self, 'in_relevant_region') else True
            old_point = batch_points[i]
            batch_points[i] = new_point
            next_density = torch.det(self._dpp_kernel(new_point))
            if torch.rand(1) < 0.5 * (next_density / curr_density).clamp(max=1):
                curr_density = next_density
            else:
                batch_points[i] = old_point
        print('  MCMC : %12d seconds to sample with %d transition' %
              (int(time.time() - mcmc_start_time), self.mcmc_step))
        return batch_points

    def __call__(self) -> Tuple[Tensor, Tensor, str]:
        points_in_batch, acq_value, greedy_optim_info_str = self.greedy_optimization()
        acq_value_list = [acq_value]
        acq_info_str_list = [greedy_optim_info_str]
        for _ in range(1, self.batch_size):
            new_point, acq_value, greedy_optim_info_str = self.greedy_optimization(points_in_batch=points_in_batch)
            acq_value_list.append(acq_value)
            acq_info_str_list.append(greedy_optim_info_str)
            points_in_batch = torch.cat([points_in_batch, new_point], dim=0)
        # MCMC begins with greedy initialization as in "Monte Carlo Markov Chain Algorithms
        # for Sampling Strongly Rayleigh Distributions and Determinantal Point Processes"
        if self.mcmc_step is not None and self.mcmc_step > 0:
            points_in_batch = self._dpp_mcmc(points_in_batch)

        acq_values = torch.cat(acq_value_list)
        acq_info_str_list.append('Acquisition values at points maximized with a different objective :'
                                 + ' '.join(['%+.2E' % acq_values[i].item() for i in range(self.batch_size)]))
        acq_info_str_list.append('Points in the batch')
        for b in range(self.batch_size):
            acq_info_str_list.append(('%2d : ' % (b + 1)) + ', '.join(['%2d' % points_in_batch[b, i].item()
                                                                       for i in range(points_in_batch.size()[1])]))

        return points_in_batch, acq_values, '\n'.join(acq_info_str_list)


def mc_batch_acquisition_objective(acquisition_samples: List[BatchAcquisition],
                                   points_in_batch: Optional[Tensor] = None):
    greedy_batch_objective_samples = [elm.greedy_batch_objective(points_in_batch) for elm in acquisition_samples]
    def mc_estimate(x):
        target_samples = []
        for greedy_batch_objective in greedy_batch_objective_samples:
            target_samples.append(greedy_batch_objective(x))
        return torch.stack(target_samples).mean(dim=0)
    return mc_estimate


def mc_acquisition(acquisition_samples: List[BatchAcquisition]):
    def mc_estimate(x):
        target_samples = []
        for acquisition_sample in acquisition_samples:
            target_samples.append(acquisition_sample.acquisition_value(x))
        return torch.stack(target_samples).mean(dim=0)
    return mc_estimate


def mc_dpp_greedy_optimization_step(acquisition_samples: List[BatchAcquisition],
                                    points_in_batch: Optional[Tensor] = None):
    acquisition_sample = acquisition_samples[0]
    search_space_type = acquisition_sample.search_space_type
    batch_size = acquisition_sample.batch_size
    gp_model_sample = acquisition_sample.gp_model
    init_data_x = gp_model_sample.train_inputs[0] \
        if points_in_batch is None else torch.cat([gp_model_sample.train_inputs[0], points_in_batch], dim=0)
    adj_mat_list = gp_model_sample.covar_module.base_kernel.adj_mat_list if search_space_type == 'discrete' else None

    acquisition_value = mc_acquisition(acquisition_samples=acquisition_samples)
    hill_climbing_objective = mc_batch_acquisition_objective(acquisition_samples=acquisition_samples,
                                                             points_in_batch=points_in_batch)

    best_input, best_output, best_ind, init_points, points_to_avoid = hill_climbing(
        search_space_type=search_space_type, objective=hill_climbing_objective,
        init_data_x=init_data_x, batch_size=batch_size, constraint=None,
        adj_mat_list=adj_mat_list if search_space_type == 'discrete' else None)

    best_input_acq_value = acquisition_value(best_input.view(1, -1))
    greedy_optim_info_str = optimization_summary(
        acquisition_value, points_in_batch=points_in_batch, init_points=init_points,
        init_point_obj_values=hill_climbing_objective(init_points).view(-1),
        best_input_obj_value=best_output, best_input_acq_value=best_input_acq_value, best_ind=best_ind)

    return best_input.view(1, -1), best_input_acq_value, greedy_optim_info_str


def mc_dpp_batch(acquisition_samples: List[BatchAcquisition]):
    batch_size = acquisition_samples[0].batch_size
    mcmc_step = acquisition_samples[0].mcmc_step
    points_in_batch, acq_value, greedy_optim_info_str = mc_dpp_greedy_optimization_step(
        acquisition_samples=acquisition_samples)
    acq_value_list = [acq_value]
    acq_info_str_list = [greedy_optim_info_str]
    for _ in range(1, batch_size):
        new_point, acq_value, greedy_optim_info_str = mc_dpp_greedy_optimization_step(
            acquisition_samples=acquisition_samples, points_in_batch=points_in_batch)
        acq_value_list.append(acq_value)
        acq_info_str_list.append(greedy_optim_info_str)
        points_in_batch = torch.cat([points_in_batch, new_point], dim=0)
    if mcmc_step is not None and mcmc_step > 0:
        raise NotImplementedError

    acq_values = torch.cat(acq_value_list)
    acq_info_str_list.append('Acquisition values at points maximized with a different objective :'
                             + ' '.join(['%+.2E' % acq_values[i].item() for i in range(batch_size)]))
    acq_info_str_list.append('Points in the batch')
    for b in range(batch_size):
        acq_info_str_list.append(('%2d : ' % (b + 1)) + ', '.join(['%2d' % points_in_batch[b, i].item()
                                                                   for i in range(points_in_batch.size()[1])]))

    return points_in_batch, acq_values, '\n'.join(acq_info_str_list)


class BUCBBatchAcquisition(BatchAcquisition):
    def __init__(self, gp_model: ExactGPRegression, data_x: Tensor, data_y: Tensor, batch_size: int,
                 maximize: bool = False, **kwargs):
        kwargs['beta'] = ucb_beta(t=self.n_data, size=self.space_size)
        super().__init__(gp_model=gp_model, data_x=data_x, data_y=data_y, batch_size=batch_size,
                         maximize=maximize, **kwargs)
        self.acq_function = confidence_bound

    def greedy_batch_objective(self, points_in_batch: Optional[Tensor] = None) -> Callable:
        if points_in_batch is None:
            points_in_batch = self.data_x.new_empty((0, self.data_x.size()[1])).long()
            hill_climbing_objective = self.acquisition_value
        else:
            beta = ucb_beta(t=self.n_data + points_in_batch.size()[0], size=self.space_size)
            acc_gram_batch = self._kernel_computation(points_in_batch).detach()
            acc_gram_chol_lower = psd_safe_cholesky(acc_gram_batch, upper=False)
            acc_gram_chol_inverse = torch.triangular_solve(
                input=torch.eye(acc_gram_batch.size(0), device=acc_gram_batch.device),
                A=acc_gram_chol_lower, upper=False)[0]

            def hill_climbing_objective(x):
                pred_mean, _ = pred_mean_std(
                    x=x, gp_model=self.gp_model, data_x=self.data_x, data_y=self.data_y,
                    mean_train=self.mean_train, cholesky_lower_inv=self.cholesky_lower_inv)
                mat = torch.mm(acc_gram_chol_inverse, self._kernel_computation(points_in_batch, x))
                posterior_var = self._kernel_computation(x, diag=True) - torch.sum(mat ** 2, dim=0)
                return pred_mean + (beta * posterior_var) ** 0.5 * (1 if self.maximize else -1)
        return hill_climbing_objective


class qPointBatchAcquisition(BatchAcquisition):
    def __init__(self, gp_model: ExactGPRegression, data_x: Tensor, data_y: Tensor, batch_size: int,
                 acq_function: Callable, maximize: bool = False, **kwargs):
        super().__init__(gp_model=gp_model, data_x=data_x, data_y=data_y, batch_size=batch_size,
                         maximize=maximize, **kwargs)
        self.acq_function = acq_function

    def greedy_batch_objective(self, points_in_batch: Optional[Tensor] = None) -> Callable:
        if points_in_batch is None:
            hill_climbing_objective = self.acquisition_value
            mean_train = self.gp_model.mean_module(self.data_x).detach()
            cholesky_lower_inv = gram_cholesky_lower_inv(gp_model=self.gp_model, data_x=self.data_x)
            if self.acq_function.__name__ in ['optimization_as_estimation']:
                incumbent = m_hat_estimate_permutation(
                    gp_model=self.gp_model, data_x=self.data_x, data_y=self.data_y, mean_train=mean_train,
                    cholesky_lower_inv=cholesky_lower_inv, maximize=self.maximize)
                self.acq_kwargs['incumbent'] = incumbent
        else:
            fantasy_x = points_in_batch
            fantasy_y, _ = pred_mean_std(
                x=fantasy_x, gp_model=self.gp_model, data_x=self.data_x, data_y=self.data_y,
                mean_train=self.mean_train, cholesky_lower_inv=self.cholesky_lower_inv)
            x_aug = torch.cat([self.data_x, fantasy_x], dim=0)
            y_aug = torch.cat([self.data_y.view(-1, 1), fantasy_y.view(-1, 1)], dim=0)
            mean_train_aug =self.gp_model.mean_module(x_aug).detach()
            cholesky_lower_inv_aug = gram_cholesky_lower_inv(gp_model=self.gp_model, data_x=x_aug)
            acq_kwargs_aug = self.acq_kwargs.copy()
            if self.acq_function.__name__ in ['expected_improvement', 'probability_improvement']:
                assert 'incumbent' in acq_kwargs_aug
                acq_kwargs_aug['incumbent'] = torch.max(y_aug).item() if self.maximize else torch.min(y_aug).item()
            elif self.acq_function.__name__ in ['optimization_as_estimation']:
                incumbent = m_hat_estimate_permutation(
                    gp_model=self.gp_model, data_x=x_aug, data_y=y_aug, mean_train=mean_train_aug,
                    cholesky_lower_inv=cholesky_lower_inv_aug, maximize=self.maximize)
                acq_kwargs_aug['incumbent'] = incumbent
            else:
                raise NotImplementedError

            def acquisition_value_aug(x: Tensor) -> Tensor:
                # backprogpagation is not called via this class, so using detach() is fine
                pred_mean_aug, pred_std_aug = pred_mean_std(
                    x=x, gp_model=self.gp_model, data_x=x_aug, data_y=y_aug,
                    mean_train=mean_train_aug, cholesky_lower_inv=cholesky_lower_inv_aug)
                return self.acq_function(mean=pred_mean_aug, sigma=pred_std_aug, **acq_kwargs_aug)

            hill_climbing_objective = acquisition_value_aug
        return hill_climbing_objective


class BaseDPPBatchAcquisition(BatchAcquisition):
    def __init__(self, gp_model: ExactGPRegression, data_x: Tensor, data_y: Tensor, batch_size: int,
                 acq_function: Callable, use_eval_data: bool, mcmc_step: Optional[int] = None, maximize: bool = False,
                 **kwargs):
        super().__init__(gp_model=gp_model, data_x=data_x, data_y=data_y, batch_size=batch_size,
                         maximize=maximize, mcmc_step=mcmc_step, **kwargs)
        self.acq_function = acq_function
        self.use_eval_data = use_eval_data
        self.weight_function = None

    def _dpp_kernel(self, x1: Tensor, x2: Optional[Tensor] = None, diag: bool = False) -> Tensor:
        return self._kernel_computation(x1=x1, x2=x2, diag=diag, use_eval_data=self.use_eval_data)

    def greedy_batch_objective(self, points_in_batch: Optional[Tensor] = None) -> Callable:
        if points_in_batch is None:
            points_in_batch = self.data_x.new_empty((0, self.data_x.size()[1])).long()
            hill_climbing_objective = self.acquisition_value
        else:
            # when use_eval_data is True, dpp_kernel is not stationary
            dpp_gram_batch = self._dpp_kernel(points_in_batch).detach()
            dpp_gram_chol_lower = psd_safe_cholesky(dpp_gram_batch, upper=False)
            dpp_gram_chol_inverse = torch.triangular_solve(
                input=torch.eye(dpp_gram_batch.size(0), device=dpp_gram_batch.device),
                A=dpp_gram_chol_lower, upper=False)[0]

            if self.name == 'DPP':  # DPP, Kathuria et al 2016
                def hill_climbing_objective(x):
                    mat = torch.mm(dpp_gram_chol_inverse, self._dpp_kernel(points_in_batch, x))
                    posterior_var = self._dpp_kernel(x, diag=True) - torch.sum(mat ** 2, dim=0)
                    return posterior_var.clamp(min=1e-9) ** 0.5
            elif self.name == 'WDPP':  # acquisition weighted DPP
                def hill_climbing_objective(x):
                    mat = torch.mm(dpp_gram_chol_inverse, self._dpp_kernel(points_in_batch, x))
                    posterior_var = self._dpp_kernel(x, diag=True) - torch.sum(mat ** 2, dim=0)
                    return self.weight_function(self.acquisition_value(x)) * posterior_var.clamp(min=1e-9) ** 0.5
            elif self.name == 'PMDPP':  # pred mean weighted DPP, will be removed later
                def hill_climbing_objective(x):
                    pred_mean, _ = pred_mean_std(
                        x=x, gp_model=self.gp_model, data_x=self.data_x, data_y=self.data_y,
                        mean_train=self.mean_train, cholesky_lower_inv=self.cholesky_lower_inv)
                    if not self.maximize:
                        pred_mean = pred_mean * -1
                    mat = torch.mm(dpp_gram_chol_inverse, self._dpp_kernel(points_in_batch, x))
                    posterior_var = self._dpp_kernel(x, diag=True) - torch.sum(mat ** 2, dim=0)
                    return torch.exp(pred_mean) * posterior_var.clamp(min=1e-9) ** 0.5
            else:
                raise ValueError
        return hill_climbing_objective


class OriginalDPPBatchAcquisition(BaseDPPBatchAcquisition):
    def __init__(self, gp_model: ExactGPRegression, data_x: Tensor, data_y: Tensor, batch_size: int,
                 acq_function: Callable, mcmc_step: Optional[int], maximize: bool = False, **kwargs):
        super().__init__(gp_model=gp_model, data_x=data_x, data_y=data_y, batch_size=batch_size,
                         acq_function=acq_function, use_eval_data=True, mcmc_step=mcmc_step, maximize=maximize,
                         **kwargs)
        self.name = 'DPP'


class PredMeanDPPBatchAcquisition(BaseDPPBatchAcquisition):  # will be removed later
    def __init__(self, gp_model: ExactGPRegression, data_x: Tensor, data_y: Tensor, batch_size: int,
                 acq_function: Callable, use_eval_data: bool, mcmc_step: Optional[int] = None, maximize: bool = False,
                 **kwargs):
        super().__init__(gp_model=gp_model, data_x=data_x, data_y=data_y, batch_size=batch_size,
                         acq_function=acq_function, use_eval_data=use_eval_data, mcmc_step=mcmc_step,
                         maximize=maximize, **kwargs)
        self.name = 'PMDPP'


class WeightedDPPBatchAcquisition(BaseDPPBatchAcquisition):
    def __init__(self, gp_model: ExactGPRegression, data_x: Tensor, data_y: Tensor, batch_size: int,
                 acq_function: Callable, weight_function: Callable, use_eval_data: bool, mcmc_step: Optional[int],
                 maximize: bool = False, **kwargs):
        super().__init__(gp_model=gp_model, data_x=data_x, data_y=data_y, batch_size=batch_size,
                         acq_function=acq_function, use_eval_data=use_eval_data, mcmc_step=mcmc_step,
                         maximize=maximize, **kwargs)
        self.name = 'WDPP'
        self.weight_function = weight_function


class AcquisitionEnsemble(Problem):

    def __init__(self, gp_model, data_x, data_y, third_acq_type, maximize: bool = False):
        self.maximize = maximize
        self.batch_sizeatch_acquisition = BatchAcquisition(gp_model=gp_model, data_x=data_x, data_y=data_y,
                                                           batch_size=0)
        self.device = data_x.device
        self.gp_model = gp_model
        if hasattr(gp_model.covar_module.base_kernel, 'fourier_freq_list'):
            self.search_space_type = 'discrete'
            self.n_cat_list = [elm.size()[0] for elm in gp_model.covar_module.base_kernel.fourier_basis_list]
            self.adj_mat_list = gp_model.covar_module.base_kernel.adj_mat_list
            super().__init__(n_obj=3, n_var=data_x.size(1),
                             xl=np.zeros(len(self.n_cat_list)), xu=np.array([elm - 1 for elm in self.n_cat_list]),
                             elementwise_evaluation=True)
        elif hasattr(gp_model.covar_module.base_kernel, 'permutation_size'):
            self.search_space_type = 'permutation'
            super().__init__(n_obj=3, n_var=data_x.size(1), elementwise_evaluation=True)
        else:
            raise NotImplementedError
        self.data_x = data_x
        self.data_y = data_y
        self.third_acq_type = third_acq_type
        self.cholesky_lower_inv = gram_cholesky_lower_inv(gp_model=gp_model, data_x=data_x)
        self.mean_train = gp_model.mean_module(data_x).detach()
        if self.search_space_type == 'discrete':
            self.m_est = m_hat_estimate_discrete(
                gp_model=gp_model, data_x=data_x, data_y=data_y,
                mean_train=self.mean_train, cholesky_lower_inv=self.cholesky_lower_inv, maximize=maximize)
        elif self.search_space_type == 'permutation':
            self.m_est = m_hat_estimate_permutation(
                gp_model=gp_model, data_x=data_x, data_y=data_y,
                mean_train=self.mean_train, cholesky_lower_inv=self.cholesky_lower_inv, maximize=maximize)
        self.incumbent = torch.max(self.data_y).item() if self.maximize else torch.min(self.data_y).item()
        self.beta = ucb_beta(t=self.data_x.size(0), size=self.n_var)

    def _evaluate(self, x, out, *args, **kwargs):
        x = torch.Tensor(x).view(1, -1)
        pred_mean, pred_std = pred_mean_std(
            x=x, gp_model=self.gp_model, data_x=self.data_x, data_y=self.data_y,
            mean_train=self.mean_train, cholesky_lower_inv=self.cholesky_lower_inv)
        f1 = -probability_improvement(mean=pred_mean, sigma=pred_std,
                                      incumbent=self.incumbent, maximize=self.maximize).item()
        f2 = -expected_improvement(mean=pred_mean, sigma=pred_std,
                                   incumbent=self.incumbent, maximize=self.maximize).item()
        if self.third_acq_type == 'ucb':
            f3 = ((-1 if self.maximize else 1)
                  * confidence_bound(mean=pred_mean, sigma=pred_std, beta=self.beta, maximize=self.maximize).item())
        elif self.third_acq_type == 'est':
            f3 = -optimization_as_estimation(mean=pred_mean, sigma=pred_std,
                                             incumbent=self.m_est, maximize=self.maximize).item()
        else:
            raise NotImplementedError

        out["F"] = [f1, f2, f3]


class MultiObjectiveEnsembleBatchAcquisition(object):
    def __init__(self, gp_model: ExactGPRegression, data_x: Tensor, data_y: Tensor, batch_size: int, maximize: bool,
                 third_acq_type: str, pop_size: int = 500, n_offsprings: int = 200):
        self.gp_model = gp_model

        if hasattr(gp_model.covar_module.base_kernel, 'fourier_freq_list'):
            self.search_space_type = 'discrete'
            self.n_cat_list = [elm.size()[0] for elm in gp_model.covar_module.base_kernel.fourier_basis_list]
            self.adj_mat_list = gp_model.covar_module.base_kernel.adj_mat_list
            self.n_data, self.n_cats = data_x.size()
            self.p_size = None
            self.space_size = np.prod(self.n_cat_list)
        elif hasattr(gp_model.covar_module.base_kernel, 'permutation_size'):
            self.search_space_type = 'permutation'
            self.n_data, self.p_size = data_x.size()
            self.n_cats = None
            self.space_size = math.factorial(self.p_size)
        else:
            raise NotImplementedError
        self.data_x = data_x
        self.data_y = data_y
        self.batch_size = batch_size
        assert third_acq_type in ['est', 'ucb']
        self.third_acq_type = third_acq_type
        self.maximize = maximize
        self.pop_size = pop_size
        self.n_offsprings = n_offsprings
        self.device = data_x.device

    def __call__(self):
        problem = AcquisitionEnsemble(gp_model=self.gp_model, data_x=self.data_x, data_y=self.data_y,
                                      third_acq_type=self.third_acq_type, maximize=self.maximize)
        if self.search_space_type == 'discrete':
            n_data, n_dim = self.data_x.size()
            init_pop = np.zeros((self.pop_size, n_dim))
            init_pop[:min(n_data, self.pop_size)] = self.data_x[torch.randperm(n_data)][:min(n_data, self.pop_size)]
            init_pop[min(n_data, self.pop_size):] = \
                random_discrete(n_cat_list=self.n_cat_list, n=max(0, self.pop_size-n_data), device=self.data_x.device)
            algorithm = NSGA2(pop_size=self.pop_size, n_offsprings=self.n_offsprings,
                              sampling=init_pop.astype(int),
                              selection=get_selection("random"),
                              crossover=get_crossover("int_sbx", prob=1.0, eta=3.0),
                              mutation=get_mutation("int_pm", eta=3.0),
                              eliminate_duplicates=True)
        elif self.search_space_type == 'permutation':
            n_data, p_size = self.data_x.size()
            init_pop = np.zeros((self.pop_size, p_size))
            init_pop[:min(n_data, self.pop_size)] = self.data_x[torch.randperm(n_data)][:min(n_data, self.pop_size)]
            for i in range(n_data, self.pop_size):
                init_pop[i] = np.random.permutation(p_size)
            algorithm = NSGA2(pop_size=self.pop_size, n_offsprings=self.n_offsprings,
                              sampling=init_pop.astype(int),
                              selection=get_selection("random"),
                              crossover=get_crossover("perm_erx"),
                              mutation=get_mutation("perm_inv"),
                              eliminate_duplicates=True)
        else:
            raise NotImplementedError

        termination = MultiObjectiveDefaultTermination(nth_gen=10, n_last=20)
        ga_start_time = time.time()
        res = minimize(problem=problem, algorithm=algorithm, termination=termination)
        batch = torch.LongTensor(res.X)
        n_dominant_samples = res.X.shape[0]
        acquisition_values = torch.FloatTensor(-res.F)
        if n_dominant_samples < self.batch_size:
            if self.search_space_type == 'discrete':
                for _ in range(res.X.shape[0], self.batch_size):
                    new_point = random_discrete(n_cat_list=self.n_cat_list, n=1, device=self.device)
                    while torch.any(torch.all(torch.eq(torch.cat([self.data_x, batch]), new_point), dim=1)):
                        new_point = random_discrete(n_cat_list=self.n_cat_list, n=1, device=self.device)
                    batch = torch.cat([batch, random_discrete(n_cat_list=self.n_cat_list,
                                                              n=1, device=self.device).view(1, -1)])
            elif self.search_space_type == 'permutation':
                for _ in range(res.X.shape[0], self.batch_size):
                    new_point = torch.randperm(self.p_size).view(1, -1)
                    while torch.any(torch.all(torch.eq(torch.cat([self.data_x, batch]), new_point), dim=1)):
                        new_point = torch.randperm(self.p_size).view(1, -1)
                    batch = torch.cat([batch, torch.randperm(n=self.p_size).view(1, -1)])
            else:
                raise NotImplementedError
            acquisition_values = \
                torch.cat([acquisition_values,
                           torch.zeros(self.batch_size - n_dominant_samples, problem.n_obj) / 0], dim=0)
        else:
            idx = torch.randperm(n=n_dominant_samples)[:self.batch_size]
            batch = batch[idx]
            acquisition_values = acquisition_values[idx]

        already_in = torch.any(torch.all(torch.eq(batch.view(self.batch_size, 1, -1), self.data_x.view(1, n_data, -1)),
                                         dim=-1), dim=1)
        if torch.any(already_in):
            not_in = torch.logical_not(already_in)
            batch = batch[not_in]
            acquisition_values = acquisition_values[not_in]
            if self.search_space_type == 'discrete':
                for i in range(batch.size()[0], self.batch_size):
                    new_point = random_discrete(n_cat_list=self.n_cat_list, n=1, device=self.data_x.device)
                    while torch.any(torch.all(torch.eq(torch.cat([self.data_x, batch]), new_point), dim=1)):
                        new_point = random_discrete(n_cat_list=self.n_cat_list, n=1, device=self.data_x.device)
                    batch = torch.cat([batch, new_point], dim=0)
                    acquisition_values = torch.cat([acquisition_values, torch.zeros(1, problem.n_obj) / 0], dim=0)
            elif self.search_space_type == 'permutation':
                for i in range(batch.size()[0], self.batch_size):
                    new_point = torch.randperm(self.p_size).view(1, -1)
                    while torch.any(torch.all(torch.eq(torch.cat([self.data_x, batch]), new_point), dim=1)):
                        new_point = torch.randperm(self.p_size).view(1, -1)
                    batch = torch.cat([batch, new_point], dim=0)
                    acquisition_values = torch.cat([acquisition_values, torch.zeros(1, problem.n_obj) / 0], dim=0)
            else:
                raise NotImplementedError

        acq_info_str_list = ['Points in the batch']
        for b in range(self.batch_size):
            acq_info_str_list.append(('%2d : ' % (b + 1)) + ', '.join(['%2d' % batch[b, i].item()
                                                                       for i in range(batch.size()[1])]))
        acq_info_str_list.append('Randomly chosen from %d dominants points' % n_dominant_samples)
        acq_info_str_list.append((('%1d acquisition functions--' % problem.n_obj)
                                  + '  '.join(['  point %d' % elm for elm in range(self.batch_size)])))
        for a in range(problem.n_obj):
            acq_info_str_list.append(
                ('Acquisition function %1d : ' % (a + 1)) + ', '.join([
                    ('   random' if torch.isnan(acquisition_values[elm, a])
                     else '%+8.2E' % acquisition_values[elm, a].item()) for elm in range(self.batch_size)]))

        acq_info_str_list.append('%d seconds to have %d dominant samples with the population of '
                                 '%d points and %d offsprings' %
                                 (int(time.time() - ga_start_time), n_dominant_samples,
                                  self.pop_size, self.n_offsprings))
        return batch, acquisition_values, '\n'.join(acq_info_str_list)


