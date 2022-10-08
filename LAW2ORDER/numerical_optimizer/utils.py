from typing import Optional, List

from itertools import combinations

import numpy as np
import torch


OPTIM_N_INITS = 30


def random_permutations(p_size, n_perm, seed: Optional[int] = None):
    data_x = torch.zeros((n_perm, p_size)).long()
    seed_list = np.random.RandomState(seed=seed).randint(0, 10000, (n_perm,))
    for i, s in enumerate(seed_list):
        data_x[i] = torch.from_numpy(np.random.RandomState(seed_list[i]).permutation(p_size))
    return data_x


def discrete_neighbors_function(adj_mat_list: List[torch.Tensor]):
    def neighbors_function(x):
        if x.dim() == 2:
            assert x.size()[0] == 1
            x = x.view(1)
        nbd_list = []
        for i, adj_mat in enumerate(adj_mat_list):
            nbds = torch.where(adj_mat[x[i]])[0]
            if nbds.numel() > 0:
                neighbor_group_i = x.repeat(nbds.numel(), 1)
                neighbor_group_i[:, i] = nbds
                nbd_list.append(neighbor_group_i)
        return torch.cat(nbd_list, dim=0)
    return neighbors_function


def permutation_neighbors(permutation):
    assert permutation.ndim == 1 or permutation.size(0) == 1
    swap_inds = torch.tensor(list(combinations(range(permutation.numel()), 2)), dtype=torch.long,
                             device=permutation.device)
    neighboring_permutations = permutation.view(1, -1).repeat(swap_inds.size(0), 1)
    inds = torch.arange(swap_inds.size(0), dtype=torch.long, device=permutation.device)
    tmp = neighboring_permutations[inds, swap_inds[:, 0]].clone()
    neighboring_permutations[inds, swap_inds[:, 0]] = neighboring_permutations[inds, swap_inds[:, 1]].clone()
    neighboring_permutations[inds, swap_inds[:, 1]] = tmp
    return neighboring_permutations


def permutation_pair_neighbors(permutation_pair):
    if permutation_pair.ndim == 1:
        permutation_pair = permutation_pair.view(1, -1)
    p_size = permutation_pair.numel() // 2
    permutation1 = permutation_pair[..., :p_size]
    permutation2 = permutation_pair[..., p_size:]
    partial_neighbor1 = permutation_neighbors(permutation1)
    partial_neighbor2 = permutation_neighbors(permutation2)
    neighbor1 = torch.cat([partial_neighbor1,
                           permutation_pair.view(1, -1)[p_size:].repeat(partial_neighbor1.size(0), 1)], dim=1)
    neighbor2 = torch.cat([permutation_pair.view(1, -1)[:p_size].repeat(partial_neighbor2.size(0), 1),
                           partial_neighbor2], dim=1)
    return torch.cat([neighbor1, neighbor2], dim=0)