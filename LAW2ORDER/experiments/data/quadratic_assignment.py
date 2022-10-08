from typing import Tuple

import numpy as np

import torch

from LAW2ORDER.experiments.config_file_path import qap_benchmark_filepath


def parse_qap_benchmark(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    filapath = qap_benchmark_filepath(name)
    with open(filapath, 'rt') as f:
        lines = f.readlines()
    i = 0
    p_size = int(lines[i])
    i += 1
    while lines[i].strip() == '':
        i += 1

    row_cnt = 0
    weight_matrix = np.empty((p_size, p_size))
    while row_cnt < p_size:
        row = [float(elm) for elm in lines[i].split()]
        i += 1
        while len(row) < p_size:
            row += [float(elm) for elm in lines[i].split()]
            i += 1
        assert len(row) == p_size
        weight_matrix[row_cnt] = row
        row_cnt += 1

    while lines[i].strip() == '':
        i += 1

    row_cnt = 0
    distance_matrix = np.empty((p_size, p_size))
    while row_cnt < p_size:
        row = [float(elm) for elm in lines[i].split()]
        i += 1
        while len(row) < p_size:
            row += [float(elm) for elm in lines[i].split()]
            i += 1
        assert len(row) == p_size
        distance_matrix[row_cnt] = row
        row_cnt += 1

    return torch.from_numpy(weight_matrix.astype(np.float32)), torch.from_numpy(distance_matrix.astype(np.float32))


def evaluate_qap_cost(permutation: torch.Tensor, weight_matrix: torch.Tensor, distance_matrix: torch.Tensor) \
        -> torch.Tensor:
    if permutation.ndim == 1:
        permutation = permutation.view(1, -1)
    assert permutation.ndim == 2

    n_p, p_size = permutation.size()
    assigment_cost = permutation.new_empty(n_p).float()

    block_size = 100
    ind1 = torch.ger(torch.arange(block_size), torch.ones(p_size).long())
    ind2 = torch.ger(torch.ones(block_size).long(), torch.arange(p_size))
    wgh_mat_rep = weight_matrix.repeat(block_size, 1, 1)
    dst_transpose_mat_rep = distance_matrix.t().repeat(block_size, 1, 1)

    i = 0
    while i + block_size < n_p:
        block_permutation_mat = permutation.new_zeros(block_size, p_size, p_size).float()
        block_permutation_mat[ind1, ind2, permutation[i:i+block_size]] = 1
        block_cost = torch.diagonal(torch.bmm(torch.bmm(wgh_mat_rep, block_permutation_mat),
                                              torch.bmm(dst_transpose_mat_rep, block_permutation_mat.permute(0, 2, 1))),
                                    dim1=-2, dim2=-1).sum(-1)
        assigment_cost[i:i + block_size] = block_cost
        i += block_size

    block_size = n_p - i
    ind1 = torch.ger(torch.arange(block_size), torch.ones(p_size).long())
    ind2 = torch.ger(torch.ones(block_size).long(), torch.arange(p_size))
    wgh_mat_rep = weight_matrix.repeat(block_size, 1, 1)
    dst_transpose_mat_rep = distance_matrix.t().repeat(block_size, 1, 1)
    block_permutation_mat = permutation.new_zeros(block_size, p_size, p_size).float()
    block_permutation_mat[ind1, ind2, permutation[i:n_p]] = 1
    block_cost = torch.diagonal(torch.bmm(torch.bmm(wgh_mat_rep, block_permutation_mat),
                                          torch.bmm(dst_transpose_mat_rep, block_permutation_mat.permute(0, 2, 1))),
                                dim1=-2, dim2=-1).sum(-1)
    assigment_cost[i:n_p] = block_cost

    return assigment_cost


def _check_evaluate_qap(permutation: torch.Tensor, weight_matrix: torch.Tensor, distance_matrix: torch.Tensor):
    n_p, p_size = permutation.size()
    cost = evaluate_qap_cost(permutation=permutation, weight_matrix=weight_matrix, distance_matrix=distance_matrix)
    for _ in range(100):
        i = torch.randint(n_p, (1,)).item()
        p = permutation[i]
        p_mat = torch.zeros(p_size, p_size)
        p_mat[torch.arange(p_size), p] = 1
        c1 = torch.sum(weight_matrix * distance_matrix[p][:, p])
        c2 = torch.trace(torch.mm(weight_matrix, torch.mm(p_mat, torch.mm(distance_matrix.t(), p_mat.t()))))
        # If distance_matrix is assumed to be symmetric, then it recovers the formula in wiki
        assert torch.isclose(c2, cost[i]) and torch.isclose(c2, evaluate_qap_cost(p, weight_matrix, distance_matrix))
        assert torch.isclose(c1, c2)


class QAP(object):
    def __init__(self, name: str):
        self.name = 'QAP-%s' % name
        self.weight_matrix, self.distance_matrix = parse_qap_benchmark(name)
        self.permutation_size = self.weight_matrix.size()[0]
        self.maximize = False
        self.n_cores = 1

    def __call__(self, x):
        return evaluate_qap_cost(x, weight_matrix=self.weight_matrix, distance_matrix=self.distance_matrix)


if __name__ == '__main__':
    #     bur26a   26
    #     bur26b   26
    #     bur26c   26
    #     bur26d   26
    #     bur26e   26
    #     bur26f   26
    #     bur26g   26
    #     bur26h   26
    #     chr12a   12
    #     chr12b   12
    #     chr12c   12
    #     chr15a   15
    #     chr15b   15
    #     chr15c   15
    #     chr18a   18
    #     chr18b   18
    #     chr20a   20
    #     chr20b   20
    #     chr20c   20
    #     chr22a   22
    #     chr22b   22
    #     chr25a   25
    #      els19   19
    #     esc128  128
    #     esc16a   16
    #     esc16b   16
    #     esc16c   16
    #     esc16d   16
    #     esc16e   16
    #     esc16f   16
    #     esc16g   16
    #     esc16h   16
    #     esc16i   16
    #     esc16j   16
    #     esc32a   32
    #     esc32b   32
    #     esc32c   32
    #     esc32d   32
    #     esc32e   32
    #     esc32g   32
    #     esc32h   32
    #     esc64a   64
    #      had12   12
    #      had14   14
    #      had16   16
    #      had18   18
    #      had20   20
    #     kra30a   30
    #     kra30b   30
    #      kra32   32
    #    lipa20a   20
    #    lipa20b   20
    #    lipa30a   30
    #    lipa30b   30
    #    lipa40a   40
    #    lipa40b   40
    #    lipa50a   50
    #    lipa50b   50
    #    lipa60a   60
    #    lipa60b   60
    #    lipa70a   70
    #    lipa70b   70
    #    lipa80a   80
    #    lipa80b   80
    #    lipa90a   90
    #    lipa90b   90
    #      nug12   12
    #      nug14   14
    #      nug15   15
    #     nug16a   16
    #     nug16b   16
    #      nug17   17
    #      nug18   18
    #      nug20   20
    #      nug21   21
    #      nug22   22
    #      nug24   24
    #      nug25   25
    #      nug27   27
    #      nug28   28
    #      nug30   30
    #      rou12   12
    #      rou15   15
    #      rou20   20
    #      scr12   12
    #      scr15   15
    #      scr20   20
    #    sko100a  100
    #    sko100b  100
    #    sko100c  100
    #    sko100d  100
    #    sko100e  100
    #    sko100f  100
    #      sko42   42
    #      sko49   49
    #      sko56   56
    #      sko64   64
    #      sko72   72
    #      sko81   81
    #      sko90   90
    #     ste36a   36
    #     ste36b   36
    #     ste36c   36
    #    tai100a  100
    #    tai100b  100
    #     tai10a   10
    #     tai10b   10
    #     tai12a   12
    #     tai12b   12
    #    tai150b  150
    #     tai15a   15
    #     tai15b   15
    #     tai17a   17
    #     tai20a   20
    #     tai20b   20
    #    tai256c  256
    #     tai25a   25
    #     tai25b   25
    #     tai30a   30
    #     tai30b   30
    #     tai35a   35
    #     tai35b   35
    #     tai40a   40
    #     tai40b   40
    #     tai50a   50
    #     tai50b   50
    #     tai60a   60
    #     tai60b   60
    #     tai64c   64
    #     tai80a   80
    #     tai80b   80
    #     tho150  150
    #      tho30   30
    #      tho40   40
    #     wil100  100
    #      wil50   50

    _name, _p_size = 'tai10a', 10
    _w_mat, _d_mat = parse_qap_benchmark(name=_name)
    _n_p = 1000
    _permutation = torch.empty(_n_p, _p_size).long()
    _check_evaluate_qap(_permutation, _w_mat, _d_mat)