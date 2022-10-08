from typing import Callable, Optional

import math

import torch
from torch import Tensor
from torch.distributions import Normal

from gpytorch.utils.cholesky import psd_safe_cholesky

from LAW2ORDER.surrogate.gp_models import ExactGPRegression


def expected_improvement(mean, sigma, incumbent: float, maximize: bool = False) -> Tensor:
    u = (mean - incumbent) / sigma
    if not maximize:
        u = -u
    normal = Normal(torch.zeros_like(u), torch.ones_like(u))
    ucdf = normal.cdf(u)
    updf = torch.exp(normal.log_prob(u))
    ei = sigma * (updf + u * ucdf)
    return ei


def confidence_bound(mean: Tensor, sigma: Tensor, beta: float, maximize: bool = False) -> Tensor:
    assert beta > 0
    return mean + beta * (1 if maximize else -1) * sigma


def probability_improvement(mean, sigma, incumbent: float, maximize: bool = False) -> Tensor:
    normal = Normal(torch.zeros_like(mean), torch.ones_like(mean))
    bound = (incumbent - mean) / sigma
    return normal.cdf(bound * (-1 if maximize else 1))


def ucb_beta(t, size, delta=0.05):
    # From the finite case
    # in Information-theoretic regret bounds for gaussian process optimization in the bandit setting
    return 2 * math.log(size * (math.pi * t) ** 2 / (6 * delta))


def optimization_as_estimation(mean, sigma, incumbent: float, maximize: bool = False) -> Tensor:
    if maximize:
        return (mean - incumbent) / sigma
    else:
        return (incumbent - mean) / sigma


def normal_cdf_approximation(x):
    b0 = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429
    t = 1 / (1 + b0 * torch.abs(x))
    p = 1 - torch.exp(-0.5 * x ** 2) / (2 * math.pi) ** 0.5 \
        * (b1 * t + b2 * t ** 2 + b3 * t ** 3 + b4 * t ** 4 + b5 * t ** 5)
    p[x < 0] = 1 - p[x < 0]
    return p


def numerical_integration(mean: Tensor, sigma: Tensor, noise_var: Tensor, m0: float, dw: float = 0.005):
    # Assuming estimating maximum
    normal = Normal(torch.zeros_like(mean), torch.ones_like(mean))
    w = m0
    m_hat = m0

    # preprocessing for numerical stability
    # sigma = sigma + noise_var ** 0.5
    if torch.max(sigma) < 1e-2:
        print('All std. dev. are too small.')
        print('Std. Dev. %.4E ~ %.4E' % (torch.min(sigma).item(), torch.max(sigma).item()))
        sigma += 1e-2
    cdf_val = normal.cdf((w - mean) / sigma)
    min_cdf = torch.min(cdf_val)
    if min_cdf == 0:  # to handl# e a special case
        w_lower_begin = float(-5.5 * torch.max(sigma) + torch.min(mean))
        w_upper_begin = float(5.5 * torch.max(sigma) + torch.max(mean))
        w_len = w_upper_begin - w_lower_begin
        while torch.min(normal.cdf((w_upper_begin - mean) / sigma)) < 1:
            w_upper_begin += w_len
        print('There exists zero cdf value')
        print('Std. Dev. %.4E ~ %.4E' % (torch.min(sigma).item(), torch.max(sigma).item()))
        print('Norm.CDF  %.8E ~ %.8E / %.2f' % (
            torch.min(cdf_val[cdf_val != 0]).item(), torch.max(cdf_val[cdf_val != 1]).item(),
            torch.logical_and(cdf_val > 0, cdf_val < 1).float().mean() * 100))
        w_lower_bound = w_lower_begin
        w_upper_bound = w_upper_begin
        while w_upper_bound - w_lower_bound > 1e-6:
            w_mid = (w_lower_bound + w_upper_bound) / 2
            cdf_val = normal.cdf((w_mid - mean) / sigma)
            min_cdf = torch.min(cdf_val).item()
            if min_cdf == 1:
                w_upper_bound = w_mid
            else:
                w_lower_bound = w_mid
        w_all_one_minimum = w_upper_bound
        assert torch.min(normal.cdf((w_all_one_minimum - mean) / sigma)) == 1
        w_lower_bound = w_lower_begin
        w_upper_bound = w_upper_begin
        while w_upper_bound - w_lower_bound > 1e-6:
            w_mid = (w_lower_bound + w_upper_bound) / 2
            cdf_val = normal.cdf((w_mid - mean) / sigma)
            min_cdf = torch.min(cdf_val).item()
            if min_cdf == 0:
                w_lower_bound = w_mid
            else:
                w_upper_bound = w_mid
        w_no_zero_minimum = w_upper_bound
        assert w_all_one_minimum >= w_no_zero_minimum
        assert torch.min(normal.cdf((w_no_zero_minimum - mean) / sigma)) > 0
        w = w_no_zero_minimum
        m_hat = w_no_zero_minimum
        dw = (w_all_one_minimum - w_no_zero_minimum) / 100
        if w + dw == w:
            print('dw %.f8 is too small' % dw)
        while w + dw == w:
            dw *= 1.5
        dw = min(dw, 0.005)
        print('%.8E ~ %.8E, dw : %.8f' % (w_no_zero_minimum, w_all_one_minimum, dw))

    cnt = 0
    prev_logprodphi = -1  # any negative value is OK to begin with
    while prev_logprodphi < 0:  # logprodphi becomes NUMERICALLY zero due to numerical capability, this stops.
        logprodphi = torch.sum(torch.log(normal.cdf((w - mean) / sigma))).clamp(min=-30)  # for numerical stability
        assert not torch.isinf(logprodphi)
        m_hat = m_hat + (1 - torch.exp(logprodphi)).item() * dw
        w += dw
        prev_logprodphi = logprodphi
        cnt += 1
    print('In EST numerical integration %d steps' % cnt)
    maximum_est = m_hat
    return maximum_est


def random_discrete(n_cat_list, n, device):
    random_points = torch.empty(n, len(n_cat_list), device=device).long()
    for i, n_cat in enumerate(n_cat_list):
        random_points[:, i] = torch.randint(low=0, high=int(n_cat), size=(n, ))
    return random_points


def m_hat_estimate_discrete(gp_model, data_x, data_y, mean_train, cholesky_lower_inv, maximize: bool = False) -> float:
    """

    :param gp_model:
    :param data_x:
    :param data_y:
    :param mean_train:
    :param cholesky_lower_inv:
    :param maximize:
    :return:
    """
    n_samples = 100000
    n_cat_list = [elm.size()[0] for elm in gp_model.covar_module.base_kernel.fourier_basis_list]
    n_data, n_dim = data_x.size()
    # x = data_x.new_empty((n_samples, n_dim))
    # x[:n_data] = data_x
    # x[n_data:] = random_discrete(n_cat_list=n_cat_list, n=n_samples-n_data, device=x.device)
    x = random_discrete(n_cat_list=n_cat_list, n=n_samples, device=data_x.device)
    mean, sigma = pred_mean_std(x=x, gp_model=gp_model, data_x=data_x, data_y=data_y,
                                mean_train=mean_train, cholesky_lower_inv=cholesky_lower_inv)
    noise_var = sigma + gp_model.likelihood.noise.detach()

    mean = mean * (1 if maximize else -1)
    best_f = torch.max(data_y * (1 if maximize else -1)).item()

    maximum_est = numerical_integration(mean=mean, sigma=sigma, noise_var=noise_var, m0=best_f, dw=0.01)

    return maximum_est * (1 if maximize else -1)


def m_hat_estimate_permutation(gp_model, data_x, data_y, mean_train, cholesky_lower_inv, maximize: bool = False) -> float:
    """

    :param gp_model:
    :param data_x:
    :param data_y:
    :param mean_train:
    :param cholesky_lower_inv:
    :param maximize:
    :return:
    """
    n_samples = 100000
    n_data, p_size = data_x.size()
    x = data_x.new_empty((n_samples, p_size))
    x[:n_data] = data_x
    for i in range(n_data, n_samples):
        x[i] = torch.randperm(p_size, device=data_x.device)
    mean, sigma = pred_mean_std(x=x, gp_model=gp_model, data_x=data_x, data_y=data_y,
                                mean_train=mean_train, cholesky_lower_inv=cholesky_lower_inv)
    noise_var = sigma + gp_model.likelihood.noise.detach()

    mean = mean * (1 if maximize else -1)
    best_f = torch.max(data_y * (1 if maximize else -1)).item()

    maximum_est = numerical_integration(mean=mean, sigma=sigma, noise_var=noise_var, m0=best_f, dw=0.005)

    return maximum_est * (1 if maximize else -1)


def pred_mean_std(x: Tensor, gp_model, data_x: Tensor, data_y: Tensor,
                  mean_train: Tensor, cholesky_lower_inv: Tensor):
    """

    :param x: points where prediction is made
    :param gp_model:
    :param data_x:
    :param data_y:
    :param mean_train: Cached for faster computation
    :param cholesky_lower_inv: Cached for faster computation
    :return:
    """
    # backprogpagation is NOT called via this class, so using detach() is fine !!
    k_train_test = gp_model.covar_module(data_x, x).evaluate().detach()
    k_test_test_diag = gp_model.covar_module(x, diag=True).detach()
    mean_test = gp_model.mean_module(x).detach()
    chol_solve = torch.mm(cholesky_lower_inv,
                              torch.cat([k_train_test, data_y.view(-1, 1) - mean_train.view(-1, 1)], dim=1))
    pred_mean = torch.mm(chol_solve[:, -1:].t(), chol_solve[:, :-1]).view(-1) + mean_test.view(-1)
    pred_std = (k_test_test_diag.view(-1) - torch.sum(chol_solve[:, :-1] ** 2, dim=0)).clamp(min=1e-9) ** 0.5
    return pred_mean, pred_std


def gram_cholesky_lower_inv(gp_model, data_x: Tensor) -> Tensor:
    gram_matrix = gp_model.covar_module(data_x).evaluate().detach() \
                  + torch.eye(data_x.size()[0], device=data_x.device) * gp_model.likelihood.noise.detach()
    cholesky_lower = psd_safe_cholesky(gram_matrix, upper=False)
    cholesky_lower_inv = torch.triangular_solve(
        input=torch.eye(gram_matrix.size(0), device=data_x.device), A=cholesky_lower, upper=False)[0]
    return cholesky_lower_inv


class AcquisitionFunction(object):
    """
    In the original BoTorch implementation, acquisition function performs unnecessary repetition of gram matrix
    computation, in this class, such unnecessary repeated computation is replaced with cached values,
    e.g. cholesky decomposition of gram matrix
    This speeds up acquisition function optimization significantly
    """
    def __init__(self, gp_model: ExactGPRegression, acq_function: Callable, data_x: Tensor, data_y: Tensor):
        self.gp_model = gp_model
        # In acq. func. optimization, optimization is performed w.r.t input not w.r.t parameter. detach() is OK
        self.cholesky_lower_inv = gram_cholesky_lower_inv(gp_model=gp_model, data_x=data_x)
        self.data_x = data_x
        self.data_y = data_y
        self.mean_train = self.gp_model.mean_module(self.data_x).detach()
        self.acq_function = acq_function
        self.acq_func_kwargs = {}

    def __call__(self, x):
        pred_mean, pred_std = pred_mean_std(
            x=x, gp_model=self.gp_model, data_x=self.data_x, data_y=self.data_y,
            mean_train=self.mean_train, cholesky_lower_inv=self.cholesky_lower_inv)
        return self.acq_function(mean=pred_mean, sigma=pred_std, **self.acq_func_kwargs)
