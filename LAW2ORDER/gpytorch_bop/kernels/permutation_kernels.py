from typing import Optional, Callable, Union

from enum import Enum

import torch
from torch import Tensor
from gpytorch import settings
from gpytorch.lazy import LazyEvaluatedKernelTensor
from gpytorch.kernels import Kernel
from gpytorch.priors import Prior
from gpytorch.constraints import Interval, Positive


from LAW2ORDER.gpytorch_bop.kernels.permutation_distances import \
    kendall_tau_distance, hamming_distance, spearman_distance, position_distance, r_distance


class EnumPermutationKernel(Enum):
    Kendall = 'KendallKernel'
    Mallows = 'MallowsKernel'
    Hamming = 'HammingKernel'
    Spearman = 'SpearmanKernel'
    Position = 'PositionKernel'


class _PermutationKernelBase(Kernel):
    """
    Modification of base class 'Kernel' for the family of permutation kernels.
    Most of permutation kernels are defined via permutation distances  where permutation distances are kernel-parameter
    independent. Therefore, storing large distance matrix can save runtime. (self.distance_matrix)
    """
    has_lengthscale = False

    def __init__(self, permutation_size: int, batch_shape: Tensor = torch.Size([]), eps: float = 1e-6, **kwargs):
        super(_PermutationKernelBase, self).__init__()
        self.active_dims = None  # if this is None, all dims are used

        self._batch_shape = batch_shape
        self.eps: float = eps
        self.permutation_size: int = permutation_size
        self.distance_function: Optional[Callable] = None
        self.stored_distance_matrix = None
        self.upper_bound = None
        self._normalizer = 1
        
    def cache_distance_matrix(self, x1, x2, diag):  # TODO : make sure that calling this function does not cause an error
        """
        The gram matrix on training data can be computed faster by using previously computed distance_matrix on the
        training data. In Bayesian optimization, at every round with new data, self.distance_matrix should be set to
        None to compute new self.distance_matrix.
        This function assumes that if k(x1, x2) has the same dimensions as self.stored_distance_matrix, then it is the
        same. This should be used with caution!!!!!!!!
        :param x1:
        :param x2:
        :param diag:
        :return:
        """
        if x1 is None and x2 is None:
            assert self.stored_distance_matrix is not None
            distance_matrix = self.stored_distance_matrix
        elif x1 is None and x2 is not None:
            raise RuntimeError('If x2 is None, then x1 should be None')
        elif x1 is not None and x2 is None:
            if diag:
                distance_matrix = self.distance_function(x1, x1, diag=diag)
            else:
                if self.stored_distance_matrix is None:
                    self.stored_distance_matrix = self.distance_function(x1, x1, diag=diag)
                    distance_matrix = self.stored_distance_matrix
                else:
                    if x1.size()[:-1] + x1.size()[:-1] == self.stored_distance_matrix.size():
                        distance_matrix = self.stored_distance_matrix
                    else:
                        self.stored_distance_matrix = self.distance_function(x1, x1, diag=diag)
                        distance_matrix = self.stored_distance_matrix
        elif x1 is not None and x2 is not None:
            if x1.size() == x2.size() and torch.all(x1 == x2):
                if self.stored_distance_matrix is None:
                    self.stored_distance_matrix = self.distance_function(x1, x1, diag=diag)
                    distance_matrix = self.stored_distance_matrix
                else:
                    if x1.size()[:-1] + x1.size()[:-1] == self.stored_distance_matrix.size():
                        distance_matrix = self.stored_distance_matrix
                    else:
                        self.stored_distance_matrix = self.distance_function(x1, x1, diag=diag)
                        distance_matrix = self.stored_distance_matrix
            else:
                distance_matrix = self.distance_function(x1, x2, diag=diag)
        return distance_matrix

    def init_params(self):
        pass

    def __call__(self, x1=None, x2=None, diag=False, last_dim_is_batch=False, **params):
        x1_, x2_ = x1, x2

        # Give x1_ and x2_ a last dimension, if necessary
        if x1_ is not None:
            if x1_.ndimension() == 1:
                x1_ = x1_.unsqueeze(0)  # each point is a vector
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(0)  # each point is a vector
        if x1_ is not None and x2_ is not None:
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError("x1_ and x2_ must have permutations of the same size!")

        if x2_ is None:
            x2_ = x1_

        if diag:
            res = super(_PermutationKernelBase, self).__call__(x1_, x2_, diag=True,
                                                               last_dim_is_batch=last_dim_is_batch, **params)
            # Did this Kernel eat the diag option?
            # If it does not return a LazyEvaluatedKernelTensor, we can call diag on the output
            if not isinstance(res, LazyEvaluatedKernelTensor):
                if res.dim() == x1_.dim() and res.shape[-2:] == torch.Size((x1_.size(-2), x2_.size(-2))):
                    res = res.diag()
            return res

        else:
            if settings.lazily_evaluate_kernels.on() and x1_ is not None:
                res = LazyEvaluatedKernelTensor(x1_, x2_, kernel=self, last_dim_is_batch=last_dim_is_batch, **params)
            else:
                res = super(_PermutationKernelBase, self).__call__(x1_, x2_,
                                                                   last_dim_is_batch=last_dim_is_batch, **params)
            return res


class KendallKernel(_PermutationKernelBase):
    def __init__(self, permutation_size: int, batch_shape: Tensor = torch.Size([]), eps: float = 1e-6, **kwargs):
        super(KendallKernel, self).__init__(permutation_size=permutation_size, batch_shape=batch_shape, eps=eps, 
                                            **kwargs)
        self.distance_function = kendall_tau_distance
        self.upper_bound = 1

    def forward(self, x1, x2, diag=False, **params):
        distance_matrix = self.cache_distance_matrix(x1, x2, diag)
        return 1 - 4 / (self.permutation_size * (self.permutation_size - 1)) * distance_matrix
    

class _ExponentialDistanceKernel(_PermutationKernelBase):

    def __init__(self, distance_function: Callable, permutation_size: int,
                 rho_value: Union[None, float, int] = None, rho_prior: Optional[Prior] = None,
                 nu_prior: Optional[Prior] = None,
                 batch_shape: Tensor = torch.Size([]), eps: float = 1e-6, **kwargs):
        super(_ExponentialDistanceKernel, self).__init__(permutation_size=permutation_size, batch_shape=batch_shape,
                                                         eps=eps, **kwargs)
        self.distance_function: Callable = distance_function
        self._normalizer: float = 1.0
        if rho_value is not None:
            assert 0 < rho_value <= 1
        self._rho_value = rho_value

        self.register_parameter(name="raw_nu",
                                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
        if nu_prior is not None:
            self.register_prior('nu_prior', nu_prior, lambda: self.nu, lambda v: self._set_nu(v))
        self.register_constraint("raw_nu", Positive())

        if self._rho_value is None:
            self.register_parameter(name="raw_rho",
                                    parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1)))
            if rho_prior is not None:
                self.register_prior('rho_prior', rho_prior, lambda: self.rho, lambda v: self._set_rho(v))
            self.register_constraint("raw_rho", Interval(0.0, 1.0))

    @property
    def nu(self) -> torch.Tensor:
        return self.raw_nu_constraint.transform(self.raw_nu)

    @nu.setter
    def nu(self, value: torch.Tensor) -> None:
        self._set_nu(value)

    def _set_nu(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_nu)
        self.initialize(raw_nu=self.raw_nu_constraint.inverse_transform(value))

    @property
    def rho(self) -> torch.Tensor:
        if self._rho_value is None:
            return self.raw_rho_constraint.transform(self.raw_rho)
        else:
            return torch.ones_like(self.raw_nu) * self._rho_value

    @rho.setter
    def rho(self, value: torch.Tensor) -> None:
        if self._rho_value is None:
            self._set_rho(value)
        else:
            self._rho_value = value.item()

    def _set_rho(self, value: torch.Tensor) -> None:
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_rho)
        self.initialize(raw_rho=self.raw_rho_constraint.inverse_transform(value))

    def init_params(self):
        self.nu = torch.empty_like(self.nu).log_normal_()
        if self._rho_value is None:
            self.rho = torch.empty_like(self.rho).uniform_()

    def forward(self, x1, x2, diag=False, **params):
        distance_matrix = self.cache_distance_matrix(x1, x2, diag)
        return torch.exp(-self.nu * (distance_matrix / float(self._normalizer)) ** self.rho)


class MallowsKernel(_ExponentialDistanceKernel):
    def __init__(self, permutation_size: int,
                 rho_value: Optional[float] = None, rho_prior: Optional[Prior] = None, nu_prior: Optional[Prior] = None,
                 batch_shape: Tensor = torch.Size([]), eps: float = 1e-6, **kwargs):
        super(MallowsKernel, self).__init__(distance_function=kendall_tau_distance,
                                            permutation_size=permutation_size,
                                            rho_value=rho_value, rho_prior=rho_prior, nu_prior=nu_prior,
                                            batch_shape=batch_shape, eps=eps, **kwargs)
        self.upper_bound = 1
        self._normalizer: float = 1  # TODO : set properly


class HammingKernel(_ExponentialDistanceKernel):
    def __init__(self, permutation_size: int,
                 rho_value: Optional[float] = None, rho_prior: Optional[Prior] = None, nu_prior: Optional[Prior] = None,
                 batch_shape: Tensor = torch.Size([]), eps: float = 1e-6, **kwargs):
        super(HammingKernel, self).__init__(distance_function=hamming_distance,
                                            permutation_size=permutation_size,
                                            rho_value=rho_value, rho_prior=rho_prior, nu_prior=nu_prior,
                                            batch_shape=batch_shape, eps=eps, **kwargs)
        self.upper_bound = 1
        self._normalizer: float = float(permutation_size)  # TODO : set properly


class SpearmanKernel(_ExponentialDistanceKernel):
    def __init__(self, permutation_size: int,
                 rho_value: Optional[float] = None, rho_prior: Optional[Prior] = None, nu_prior: Optional[Prior] = None,
                 batch_shape: Tensor = torch.Size([]), eps: float = 1e-6, **kwargs):
        super(SpearmanKernel, self).__init__(distance_function=spearman_distance,
                                             permutation_size=permutation_size,
                                             rho_value=rho_value, rho_prior=rho_prior, nu_prior=nu_prior,
                                             batch_shape=batch_shape, eps=eps, **kwargs)
        self.upper_bound = 1
        self._normalizer: float = float(permutation_size)  # TODO : set properly


class PositionKernel(_ExponentialDistanceKernel):
    def __init__(self, permutation_size: int, nu_prior: Optional[Prior] = None,
                 batch_shape: Tensor = torch.Size([]), eps: float = 1e-6, **kwargs):
        super(PositionKernel, self).__init__(distance_function=position_distance,
                                             permutation_size=permutation_size,
                                             rho_value=1.0, rho_prior=None, nu_prior=nu_prior,
                                             batch_shape=batch_shape, eps=eps, **kwargs)
        self.upper_bound = 1
        self._normalizer: float = float(permutation_size)  # TODO : set properly


