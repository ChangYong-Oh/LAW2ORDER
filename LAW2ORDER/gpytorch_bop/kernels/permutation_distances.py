from typing import List, Tuple

import numpy as np
from scipy.stats import kendalltau

import torch


def hamming_distance(permutation1: torch.LongTensor, permutation2: torch.LongTensor,
                     diag: bool = False) -> torch.Tensor:
    """
    Right invariant
    :param permutation1:
    :param permutation2:
    :param diag:
    :return:
    """
    ndim1 = permutation1.ndim
    ndim2 = permutation2.ndim
    assert ndim1 > 0 and ndim2 > 0
    assert permutation1.size(-1) == permutation2.size(-1)
    if ndim1 == 1:
        permutation1 = permutation1.reshape((1, -1))
    if ndim2 == 1:
        permutation2 = permutation2.reshape((1, -1))
    p_len = permutation1.size(-1)

    if diag:
        assert permutation1.size() == permutation2.size()
        distance = torch.sum(permutation1 != permutation2, dim=-1).float()
    else:
        batch1 = list(permutation1.size()[:-1])
        batch2 = list(permutation2.size()[:-1])
        unsqueezed_size1 = batch1 + [1 for _ in range(len(batch2))] + [p_len]
        unsqueezed_size2 = [1 for _ in range(len(batch1))] + batch2 + [p_len]
        distance = torch.sum(permutation1.view(unsqueezed_size1) != permutation2.view(unsqueezed_size2), dim=-1).float()

    return distance


def spearman_distance(permutation1: torch.LongTensor, permutation2: torch.LongTensor,
                      diag: bool = False) -> torch.Tensor:
    """
    Right invariant
    :param permutation1:
    :param permutation2:
    :param diag
    :return:
    """
    ndim1 = permutation1.ndim
    ndim2 = permutation2.ndim
    assert ndim1 > 0 and ndim2 > 0
    assert permutation1.size(-1) == permutation2.size(-1)
    if ndim1 == 1:
        permutation1 = permutation1.reshape((1, -1))
    if ndim2 == 1:
        permutation2 = permutation2.reshape((1, -1))
    p_len = permutation1.size(-1)

    if diag:
        assert permutation1.size() == permutation2.size()
        distance = torch.sum(torch.abs(permutation1 - permutation2), dim=-1).float()
    else:
        batch1 = list(permutation1.size()[:-1])
        batch2 = list(permutation2.size()[:-1])
        unsqueezed_size1 = batch1 + [1 for _ in range(len(batch2))] + [p_len]
        unsqueezed_size2 = [1 for _ in range(len(batch1))] + batch2 + [p_len]
        distance = torch.sum(torch.abs(permutation1.view(unsqueezed_size1) - permutation2.view(unsqueezed_size2)),
                             dim=-1).float()

    return distance


def position_distance(permutation1: torch.LongTensor, permutation2: torch.LongTensor,
                      diag: bool = False) -> torch.Tensor:
    """
    Left invariant.
    This can be regarded as left invariant spearman distance.
    Positive semidefinite can be shown following 'Gaussian Field on the symmetric group'
    :param permutation1:
    :param permutation2:
    :param diag
    :return:
    """
    ndim1 = permutation1.ndim
    ndim2 = permutation2.ndim
    assert ndim1 > 0 and ndim2 > 0
    assert permutation1.size(-1) == permutation2.size(-1)
    if ndim1 == 1:
        permutation1 = permutation1.reshape((1, -1))
    if ndim2 == 1:
        permutation2 = permutation2.reshape((1, -1))
    p_len = permutation1.size(-1)

    if diag:
        assert permutation1.size() == permutation2.size()
        distance = torch.sum(torch.abs(torch.argsort(permutation1, dim=-1) - torch.argsort(permutation2, dim=-1)),
                             dim=-1).float()
    else:
        batch1 = list(permutation1.size()[:-1])
        batch2 = list(permutation2.size()[:-1])
        unsqueezed_size1 = batch1 + [1 for _ in range(len(batch2))] + [p_len]
        unsqueezed_size2 = [1 for _ in range(len(batch1))] + batch2 + [p_len]
        distance = torch.sum(torch.abs(torch.argsort(permutation1, dim=-1).view(unsqueezed_size1) -
                                       torch.argsort(permutation2, dim=-1).view(unsqueezed_size2)), dim=-1)

    return distance


def levenshtein_distance(permutation1: torch.LongTensor, permutation2: torch.LongTensor,
                         diag: bool = False) -> torch.Tensor:
    ndim1 = permutation1.ndim
    ndim2 = permutation2.ndim
    assert ndim1 > 0 and ndim2 > 0
    assert permutation1.size(-1) == permutation2.size(-1)
    if ndim1 == 1:
        permutation1 = permutation1.reshape((1, -1))
    if ndim2 == 1:
        permutation2 = permutation2.reshape((1, -1))
    p_len = permutation1.size(-1)

    if diag:
        assert permutation1.size() == permutation2.size()
    else:
        batch1 = list(permutation1.size()[:-1])
        batch2 = list(permutation2.size()[:-1])
        permutation1 = permutation1.view(batch1 + [1 for _ in range(len(batch2))] + [p_len])
        permutation2 = permutation2.view([1 for _ in range(len(batch1))] + batch2 + [p_len])
        dist_mat = permutation1.new_zeros(1 + p_len, 1 + p_len).float()
        dist_mat[:, 0] = torch.arange(1 + p_len, device=permutation1.device).float()
        dist_mat[0, :] = dist_mat[:, 0]
        distance = dist_mat.view([1 for _ in range(len(batch1) + len(batch2))] + [1 + p_len, 1 + p_len]
                                 ).repeat(batch1 + batch2 + [1, 1])
        for i in range(1, 2 * p_len + 1):
            ind1 = torch.arange(1, i + 1) if i <= p_len else torch.arange(i - p_len + 1, p_len + 1)
            ind2 = i + 1 - ind1
            ind1prev = ind1 - 1
            ind2prev = ind2 - 1
            distance[..., ind1, ind2] = torch.min(
                torch.min(distance[..., ind1prev, ind2], distance[..., ind1, ind2prev]) + 1,
                distance[..., ind1prev, ind2prev] + (permutation1[..., ind1prev] != permutation2[..., ind2prev]))
        return distance[..., -1, -1]


def r_distance(permutation1: torch.LongTensor, permutation2: torch.LongTensor,
                         diag: bool = False) -> torch.Tensor:
    ndim1 = permutation1.ndim
    ndim2 = permutation2.ndim
    assert ndim1 > 0 and ndim2 > 0
    assert permutation1.size(-1) == permutation2.size(-1)
    if ndim1 == 1:
        permutation1 = permutation1.reshape((1, -1))
    if ndim2 == 1:
        permutation2 = permutation2.reshape((1, -1))
    p_len = permutation1.size(-1)

    if diag:
        equal = permutation1.unsqueeze(-1) == permutation2.unsqueeze(-2)
    else:
        batch1 = list(permutation1.size()[:-1])
        batch2 = list(permutation2.size()[:-1])
        permutation1 = permutation1.view(batch1 + [1 for _ in range(len(batch2))] + [p_len, 1])
        permutation2 = permutation2.view([1 for _ in range(len(batch1))] + batch2 + [1, p_len])
        equal = permutation1 == permutation2
    distance = torch.sum(torch.any(torch.logical_and(equal[..., 1:, 1:], equal[..., :-1, :-1]), dim=-2), dim=-1)

    return distance


def kendall_tau_distance(permutation1: torch.LongTensor, permutation2: torch.LongTensor,
                         diag: bool = False) -> torch.Tensor:
    """
    Right invariant
    The code is vectorized which makes it more complicated than non-vectorized version.
    :param permutation1:
    :param permutation2:
    :param diag
    :return:
    """
    ndim1 = permutation1.ndim
    ndim2 = permutation2.ndim
    assert ndim1 > 0 and ndim2 > 0
    assert permutation1.size(-1) == permutation2.size(-1)
    if ndim1 == 1:
        permutation1 = permutation1.reshape((1, -1))
    if ndim2 == 1:
        permutation2 = permutation2.reshape((1, -1))
    p_len = permutation1.size(-1)

    if diag:
        assert permutation1.size() == permutation2.size()
        p1 = permutation1.view(-1, p_len)
        p2 = permutation2.view(-1, p_len)
        p2_p1inv = p2[[torch.arange(0, p1.size(0), device=p1.device).view(-1, 1).repeat(1, p_len),
                       torch.argsort(p1, dim=-1)]].view(permutation1.size())
    else:
        batchdim1 = ndim1 - 1
        batchdim2 = ndim2 - 1
        p2_p1inv = permutation2[[slice(None) for _ in range(batchdim2)] + [torch.argsort(permutation1, dim=-1)]].permute(
            *np.concatenate([np.arange(batchdim1) + batchdim2, np.arange(batchdim2), [batchdim1 + batchdim2]]))
    # Time complexity is asymptotic, therefore for small size permutation n2 algorithm is faster
    # GPU really helps to accelerate
    return _counting_inversion_n2(permutation=p2_p1inv)
    # return _counting_inversion_nlogn(permutation=p2_p1inv)


def _counting_inversion_nlogn(permutation: torch.LongTensor):
    batch_size = permutation.size()[:-1]
    p_len = permutation.size(-1)
    p = permutation.reshape(-1, p_len)
    _, inversion = _merge_sort_inversion(p)
    return inversion.view(batch_size)


def _merge_sort_inversion(permutation: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert permutation.ndim == 2
    n_permutations, p_len = permutation.size()
    p_ids = torch.arange(n_permutations, device=permutation.device)
    if p_len == 1:
        return permutation, permutation.new_zeros(n_permutations)
    else:
        mid = p_len // 2
        a, inversions_a = _merge_sort_inversion(
            permutation.index_select(dim=-1, index=torch.arange(0, mid, device=permutation.device)))
        b, inversions_b = _merge_sort_inversion(
            permutation.index_select(dim=-1, index=torch.arange(mid, p_len, device=permutation.device)))
        c = torch.zeros_like(permutation)

        a_ind = permutation.new_zeros(n_permutations).type_as(p_ids)
        b_ind = permutation.new_zeros(n_permutations).type_as(p_ids)
        inversions = inversions_a + inversions_b

        mask_unfinish_a: torch.BoolTensor = permutation.new_ones(n_permutations).type(torch.bool)
        mask_unfinish_b: torch.BoolTensor = permutation.new_ones(n_permutations).type(torch.bool)
        mask_unfinish = torch.logical_and(mask_unfinish_a, mask_unfinish_b)

        while torch.any(mask_unfinish):
            p_ids_unfinish = p_ids[mask_unfinish]
            mask_compare: torch.BoolTensor = \
                a[p_ids_unfinish, a_ind[p_ids_unfinish]] < b[p_ids_unfinish, b_ind[p_ids_unfinish]]

            p_ids_a = p_ids_unfinish[mask_compare]
            c[p_ids_a, a_ind[p_ids_a] + b_ind[p_ids_a]] = a[p_ids_a, a_ind[p_ids_a]]
            a_ind[p_ids_a] += 1
            mask_unfinish_a[p_ids_a[a_ind[p_ids_a] == mid]] = False

            p_ids_b = p_ids_unfinish[torch.logical_not(mask_compare)]
            c[p_ids_b, a_ind[p_ids_b] + b_ind[p_ids_b]] = b[p_ids_b, b_ind[p_ids_b]]
            b_ind[p_ids_b] += 1
            mask_unfinish_b[p_ids_b[b_ind[p_ids_b] == p_len - mid]] = False

            inversions[p_ids_b] += mid - a_ind[p_ids_b]

            mask_unfinish = torch.logical_and(mask_unfinish_a, mask_unfinish_b)

        while torch.any(mask_unfinish_a):
            p_ids_unfinish_a = p_ids[mask_unfinish_a]
            c[p_ids_unfinish_a, a_ind[p_ids_unfinish_a] + b_ind[p_ids_unfinish_a]] = \
                a[p_ids_unfinish_a, a_ind[p_ids_unfinish_a]]
            a_ind[p_ids_unfinish_a] += 1
            mask_unfinish_a[p_ids_unfinish_a[a_ind[p_ids_unfinish_a] == mid]] = False

        while torch.any(mask_unfinish_b):
            p_ids_unfinish_b = p_ids[mask_unfinish_b]
            c[p_ids_unfinish_b, a_ind[p_ids_unfinish_b] + b_ind[p_ids_unfinish_b]] = \
                b[p_ids_unfinish_b, b_ind[p_ids_unfinish_b]]
            b_ind[p_ids_unfinish_b] += 1
            mask_unfinish_b[p_ids_unfinish_b[b_ind[p_ids_unfinish_b] == p_len - mid]] = False

    return c, inversions


def _counting_inversion_n2(permutation: torch.LongTensor) -> torch.Tensor:
    """

    :param permutation: permutation.size(-1) is the length of each permutation
    :return:
    """
    batch_size = permutation.size()[:-1]
    p_len = permutation.size(-1)
    inversion_cnt = permutation.new_zeros(batch_size)
    for i in range(p_len):
        inversion_cnt += torch.sum(
            torch.index_select(permutation, dim=-1, index=torch.arange(i, i + 1, device=permutation.device)) >
            torch.index_select(permutation, dim=-1, index=torch.arange(i + 1, p_len, device=permutation.device)),
            dim=-1)
    return inversion_cnt


def left_invariant_kendall_tau_distance(permutation1: torch.LongTensor, permutation2: torch.LongTensor,
                                        diag: bool = False) -> torch.Tensor:
    """
    Left invariant
    :param permutation1:
    :param permutation2:
    :param diag:
    :return:
    """
    raise NotImplementedError
    assert permutation1.ndim == permutation2.ndim == 1
    assert permutation1.numel() == permutation2.numel()
    p_len = permutation1.numel()
    normalizer = p_len * (p_len - 1) / 2
    p = torch.argsort(permutation1)[permutation2]
    return torch.sum(torch.tril(p.view(-1, 1) > p.view(1, -1))) / normalizer


def scipy_kendall_tau(p1: torch.LongTensor, p2: torch.LongTensor):
    assert p1.numel() == p2.numel()
    return 0.5 * (1 - kendalltau(p1.numpy(), p2.numpy()).correlation)


def scipy_comparison_kendall_tau(p_len: int):
    dist_func = left_invariant_kendall_tau_distance
    test_invariance = 'left'
    assert test_invariance in ['left', 'right']
    p1 = torch.randperm(p_len)
    p2 = torch.randperm(p_len)
    p1_sortind = torch.argsort(p1)
    p2_sortind = torch.argsort(p2)
    rnd_sortind = torch.randperm(p_len)
    if test_invariance == 'right':
        p1_1 = p1[p1_sortind]
        p2_1 = p2[p1_sortind]
        p1_2 = p1[p2_sortind]
        p2_2 = p2[p2_sortind]
        p1_r = p1[rnd_sortind]
        p2_r = p2[rnd_sortind]
    elif test_invariance == 'left':
        p1_1 = p1_sortind[p1]
        p2_1 = p1_sortind[p2]
        p1_2 = p2_sortind[p1]
        p2_2 = p2_sortind[p2]
        p1_r = rnd_sortind[p1]
        p2_r = rnd_sortind[p2]
    print(p1)
    print(p2)
    print('-' * 50)
    print(p1_sortind[p1])
    print(p1_sortind[p2])
    print('-' * 50)
    print(p2_sortind[p1])
    print(p2_sortind[p2])
    print('-' * 50)
    print(rnd_sortind[p1])
    print(rnd_sortind[p2])
    print('-' * 50)
    print('%5.2f %5.2f %5.2f %5.2f' % (dist_func(p1, p2).item(),
                                       dist_func(p1_1, p2_1).item(),
                                       dist_func(p1_2, p2_2).item(),
                                       dist_func(p1_r, p2_r).item()))
    dist = []
    for elm1, elm2 in [(p1, p2), (p1_1, p2_1), (p1_2, p2_2), (p1_r, p2_r)]:
        dist.append(scipy_kendall_tau(elm1, elm2))
    print(' '.join(['%5.2f' % elm for elm in dist]))


if __name__ == '__main__':
    import time
    # scipy_comparison_kendall_tau(10)
    _p_len = 20
    _size1 = [_p_len * (_p_len - 1) // 2]
    _size2 = [1]
    _p1 = torch.ones(np.prod(_size1), _p_len).type(torch.int)
    _p2 = torch.ones(np.prod(_size2), _p_len).type(torch.int)
    for _i in range(np.prod(_size1)):
        _p1[_i] = torch.randperm(_p_len)
    for _j in range(np.prod(_size2)):
        _p2[_j] = torch.randperm(_p_len)
    # for d_func in [position_distance, hamming_distance, r_distance, levenshtein_distance]:
    #     _time_start = time.time()
    #     d_func(_p1, _p2)
    #     print('%30s : %9.6f' % (d_func.__name__, time.time() - _time_start))
    gram = levenshtein_distance(_p1, _p1) / 10
    a = torch.randn(_p1.size(0), 1)
    a -= torch.mean(a)
    print(torch.mm(torch.mm(a.t(), gram), a))
    # _time_start = time.time()
    # r_distance(_p1, _p2)
    # print(time.time() - _time_start)
    # _time_start = time.time()
    # _inv_cnt_nlogn = _counting_inversion_nlogn(_p1)
    # _time_nlogn = time.time() - _time_start
    # _time_start = time.time()
    # _inv_cnt_n2 = _counting_inversion_n2(_p1)
    # _time_n2 = time.time() - _time_start
    # assert (torch.all(_inv_cnt_nlogn == _inv_cnt_n2))
    # print(_time_nlogn)
    # print(_time_n2)


