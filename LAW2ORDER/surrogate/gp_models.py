from typing import List, Union, Dict, Optional, Tuple

import time
import numpy as np
from pathos import multiprocessing
import multiprocess.context as ctx

import torch
import gpytorch

from LAW2ORDER.gpytorch_bop.kernels import NeuralKernelNetwork
from LAW2ORDER.gpytorch_bop.kernels.permutation_kernels import _ExponentialDistanceKernel

ctx._force_start_method('spawn')
OPTIMIZE_N_CORES_USED = 16


class ExactGPRegression(gpytorch.models.ExactGP):

    _num_outputs = 1

    def __init__(self, train_x, train_y, kernel, likelihood=gpytorch.likelihoods.GaussianLikelihood()):
        super(ExactGPRegression, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def init_params(self):
        self.mean_module.constant.data = torch.empty_like(self.mean_module.constant).normal_(
            torch.mean(self.train_targets), 0.1 * torch.std(self.train_targets))
        if isinstance(self.covar_module, gpytorch.kernels.ScaleKernel):
            self.covar_module.base_kernel.init_params()
            target_var = torch.var(self.train_targets)
            lognormal_mean = torch.log(target_var * 0.9999)
            lognormal_std = (2 * (torch.log(target_var) - lognormal_mean)) ** 0.5
            self.covar_module.outputscale = torch.empty_like(self.covar_module.outputscale).log_normal_(
                lognormal_mean, lognormal_std)
        else:
            self.covar_module.init_params()
        # Variance of observational[ noise
        self.likelihood.noise = torch.exp(torch.empty_like(self.likelihood.noise_covar.noise).uniform_(-4, -2))\
                                * torch.var(self.train_targets)

    def kernel_name_str(self):
        if isinstance(self.covar_module, gpytorch.kernels.ScaleKernel):
            kernel_name_list = [self.covar_module.base_kernel.__class__.__name__]
            if isinstance(self.covar_module.base_kernel, _ExponentialDistanceKernel):
                rho_info = 'rho=' + ('TRAIN' if self.covar_module.base_kernel._rho_value is None
                                     else ('%5.3f' % self.covar_module.base_kernel._rho_value))
                kernel_name_list.append(rho_info)
            return '-'.join(kernel_name_list)
        else:
            raise NotImplementedError


def optimize_mll_parallel(train_x: torch.Tensor, train_y: torch.Tensor,
                          model_dict: Dict, mll_dict: Dict, optimizer_dict: Dict,
                          param_init_dict: Dict):
    assert len(model_dict) == len(optimizer_dict) == len(mll_dict) == len(param_init_dict)

    n_processes = max(1, multiprocessing.cpu_count() // int(OPTIMIZE_N_CORES_USED))
    pool = multiprocessing.Pool(n_processes)
    args_list = []
    for model_name in model_dict.keys():
        model = model_dict[model_name]
        optimizer = optimizer_dict[model_name]
        mll = mll_dict[model_name]
        param_init = param_init_dict[model_name]
        args_list.append((train_x, train_y, model_name, model, mll, optimizer, param_init, False))

    start_time = time.time()
    print('Negative Marginal Likelihood Minimization : '
          'Multiprocessing with %d processes for %d Random initialization' % (n_processes, len(args_list)))
    model_names, models, optimizers, mlls, losses = list(zip(*pool.starmap_async(optimize_mll, args_list).get()))
    negative_mll = dict(zip(model_names, losses))
    optimized_model_dict = dict(zip(model_names, models))
    optimizer_dict = dict(zip(model_names, optimizers))
    mll_dict = dict(zip(model_names, mlls))
    print('%12d seconds to run %d negative marginal lilkelihood minimization' %
          (time.time() - start_time, len(args_list)))

    return negative_mll, optimized_model_dict, mll_dict, optimizer_dict


def optimize_mll(train_x: torch.Tensor, train_y: torch.Tensor, model_name: str,
                 model, mll, optimizer, param_init: bool, report_progress: bool = False):
    model.train()
    if param_init:
        model.init_params()
    if report_progress:
        print(model_name, sum([p.numel() for p in model.parameters()]))
    prev_loss = np.inf
    i = 0
    prev_loss_log = np.inf
    model_state_dict = None
    optimizer_state_dict = None
    while i < 3000:
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y.view(-1))
        loss.backward()
        optimizer.step()
        if (i + 1) % 100 == 0:
            loss_log = loss.item()
            if loss_log <= prev_loss_log:
                prev_loss_log = loss_log
                model_state_dict = model.state_dict()
                optimizer_state_dict = optimizer.state_dict()
            if report_progress:
                print('\t\tIter %4d - Loss: %.8f' % (i + 1, loss_log))
        # ftol of scipy.optimize.minimize(method='L-BFGS-B')
        if np.abs((prev_loss - loss.item()) / max(abs(prev_loss), abs(loss.item()), 1)) < 1e-8:
            if report_progress:
                print('\t\tIter %4d - Loss: %.8f' % (i + 1, loss.item()))
            return model_name, model, optimizer, mll, loss.item()  # the arguments (model, optimizer, mll) are updated, this is for comparison
        prev_loss = loss.item()
        i += 1
    # When the optimization does not converge

    model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    return model_name, model, optimizer, mll, prev_loss_log


def evaluate_ll_rmse(test_x: torch.Tensor, test_y: torch.Tensor, model_dict: Dict):
    ll_rmse = dict()
    for model_name, model in model_dict.items():
        model.eval()
        f_pred = model(test_x)
        mean = f_pred.mean.view(-1)
        variance = f_pred.variance.view(-1) + model.likelihood.noise.item()
        log_likelihood = torch.mean(-0.5 * torch.log(2 * np.pi * variance)
                                    - 0.5 * (test_y.view(-1) - mean) ** 2 / variance)
        rmse = torch.mean((test_y.view(-1) - mean) ** 2) ** 0.5

        ll_rmse[model_name] = {'eval_ll': log_likelihood.item(), 'eval_rmse': rmse.item()}
    return ll_rmse


def pick_best_model(negative_mll: Dict, model_dict: Dict) -> Tuple[str, str]:
    best_model_name = min(negative_mll, key=negative_mll.get)

    info_str_list = []
    for m_name, model in model_dict.items():
        if hasattr(model.covar_module, 'outputscale'):
            outputscale = model.covar_module.outputscale.data.item()
        elif hasattr(model.covar_module, 'layer_name_format'):
            l = 1
            while hasattr(model.covar_module, model.covar_module.layer_name_format % l):
                l += 1
            outputscale = torch.sum(getattr(model.covar_module,
                                            model.covar_module.layer_name_format % (l - 1)).weight.data).item()
        else:
            raise NotImplementedError
        info_str_list.append('%s neg.mll : %12.8f / outputscale : %12.8f / noise : %12.8f / ratio : %12.8f %s'
                             % (m_name, negative_mll[m_name], outputscale, model.likelihood.noise.data.item(),
                                outputscale / model.likelihood.noise.data.item(), '++'
                                if best_model_name == m_name else ''))

    return best_model_name, '\n'.join(info_str_list)