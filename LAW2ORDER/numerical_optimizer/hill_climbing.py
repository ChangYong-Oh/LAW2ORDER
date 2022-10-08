from typing import Callable, Optional, List

import sys

import numpy as np

import torch
from torch import Tensor

from LAW2ORDER.numerical_optimizer.utils import \
    permutation_neighbors, permutation_pair_neighbors, discrete_neighbors_function


class _HillClimbingBase(object):
    def __init__(self, objective: Callable, initial_state: torch.Tensor, minimize: bool,
                 states_to_avoid: Optional[torch.Tensor] = None, constraint: Optional[Callable] = None,
                 max_update: int = 100):
        self.max_update = max_update
        self.objective = objective
        self.n_evaluations = 0
        self.minimize = minimize
        self.states_to_avoid = states_to_avoid
        self.cnt_update = 0
        self.state = initial_state
        self.neighbor_function = None
        self.constraint = constraint  # If not None, a function returning True for valid inputs

    def move(self):
        state_value = self.objective(self.state.view(1, -1)).detach().item()
        neighbors = self.neighbor_function(self.state)
        if self.states_to_avoid is not None:
            valid_idx = torch.logical_not(torch.any(torch.all(
                torch.eq(neighbors.unsqueeze(1), self.states_to_avoid.unsqueeze(0)), dim=2), dim=1))
        else:
            valid_idx = torch.ones(neighbors.size(0), device=neighbors.device).bool()
        if self.constraint is not None:
            valid_idx = torch.logical_and(valid_idx, self.constraint(neighbors).view(-1))
        if not torch.any(valid_idx):
            return False, state_value

        valid_nbd = neighbors[valid_idx]
        valid_nbd_value = self.objective(valid_nbd).detach()
        valid_best_ind = torch.argmin(valid_nbd_value) if self.minimize else torch.argmax(valid_nbd_value)
        valid_best_value = valid_nbd_value[valid_best_ind].item()
        updated = False

        # 1st condition is to handle the case that the initial point is one of the states to avoid
        if ((self.states_to_avoid is not None
             and torch.any(torch.all(torch.eq(self.state.view(1, -1), self.states_to_avoid), dim=1)))
                or (self.constraint is not None and not self.constraint(self.state.view(1, -1)))):
            self.state = valid_nbd[valid_best_ind].view(self.state.size())
            state_value = valid_best_value
            updated = True
        else:
            if self.minimize:
                if valid_best_value < state_value:
                    self.state = valid_nbd[valid_best_ind].view(self.state.size())
                    state_value = valid_best_value
                    updated = True
            else:
                if valid_best_value > state_value:
                    self.state = valid_nbd[valid_best_ind].view(self.state.size())
                    state_value = valid_best_value
                    updated = True
        return updated, state_value

    def climb(self):
        prev_state_value = np.inf if self.minimize else -np.inf
        self.cnt_update = self.max_update
        # Bounding the number of updates, the number of updates can be checked in log files
        for c in range(self.max_update):
            updated, state_value = self.move()
            # (cnt_update > 100) : To handle the case where it starts from an already evaluated point
            # with a large acquisition value
            # The others for long-lasting optimization for marginal improvement.
            if (not updated) \
                    or (state_value != 0 and abs(prev_state_value - state_value) / abs(state_value) < 1e-8) \
                    or (abs(prev_state_value - state_value) < 1e-12):
                self.cnt_update = c
                break
            prev_state_value = state_value
            # if (c + 1) % 100 == 0:
            #     sys.stdout.write('%.10f(%d) ' % (state_value, c + 1))
            #     sys.stdout.flush()
        return self.state, state_value


class DiscreteHillClimbing(_HillClimbingBase):
    def __init__(self, adj_mat_list: List[Tensor], objective: Callable, initial_state: torch.Tensor, minimize: bool,
                 states_to_avoid: Optional[torch.Tensor] = None, constraint: Optional[Callable] = None):
        super().__init__(objective=objective, initial_state=initial_state, minimize=minimize,
                         states_to_avoid=states_to_avoid, constraint=constraint, max_update=1000)
        self.neighbor_function = discrete_neighbors_function(adj_mat_list=adj_mat_list)


class PermutationHillClimbing(_HillClimbingBase):
    def __init__(self, objective: Callable, initial_state: torch.Tensor, minimize: bool,
                 states_to_avoid: Optional[torch.Tensor] = None, constraint: Optional[Callable] = None):
        super().__init__(objective=objective, initial_state=initial_state, minimize=minimize,
                         states_to_avoid=states_to_avoid, constraint=constraint)
        self.neighbor_function = permutation_neighbors


class PermutationPairHillClimbing(_HillClimbingBase):
    def __init__(self, objective: Callable, initial_state: torch.Tensor, minimize: bool,
                 states_to_avoid: Optional[torch.Tensor] = None, constraint: Optional[Callable] = None):
        super().__init__(objective=objective, initial_state=initial_state, minimize=minimize,
                         states_to_avoid=states_to_avoid, constraint=constraint)
        self.neighbor_function = permutation_pair_neighbors

