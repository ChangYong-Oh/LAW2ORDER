from typing import List

import re

import torch

from LAW2ORDER.experiments.config_file_path import fsp_benchmark_filepath


def parse_fsp_benchmark(name: str) -> torch.Tensor:
    filapath = fsp_benchmark_filepath()
    with open(filapath, 'rt') as f:
        text = f.read()
    split_ptr = r"[\s]*[+]+[\s]*\n"
    token_list = re.compile(split_ptr).split(text)[2:]
    instance_name_list = [elm.replace('instance', '').strip()
                          for elm in token_list if re.search(r"^instance[\s]+[\w]+", elm.strip())]
    if name not in instance_name_list:
        raise AssertionError('Available names are %s' % ','.join(instance_name_list))
    for i in range(len(token_list)):
        if name in token_list[i]:
            break
    data_token = token_list[i + 1]
    return _parse_token(data_token)


def _parse_token(token: List[str]) -> torch.Tensor:
    lines = token.split('\n')
    shape = [int(elm) for elm in lines[1].split()]
    data = torch.empty(shape)
    for i in range(shape[0]):
        data[i] = torch.Tensor([float(elm) for elm in lines[i + 2].split()][1::2])
    return data


def evaluate_fsp_time(permutation: torch.Tensor, processing_time_matrix: torch.Tensor) -> torch.Tensor:
    """
    Explained in 'A genetic algorithm for flowshop sequencing; Colin R.Reeves; Computers & Operations Research, 1995'
    :param permutation:
    :param processing_time_matrix:
    :return:
    """
    n_jobs, n_machines = processing_time_matrix.size()
    completion_time = processing_time_matrix[permutation]
    completion_time[:, :, 0] = torch.cumsum(completion_time[:, :, 0], dim=1)
    completion_time[:, 0, :] = torch.cumsum(completion_time[:, 0, :], dim=1)
    for i in range(1, n_jobs):
        for j in range(1, n_machines):
            completion_time[:, i, j] += torch.max(completion_time[:, i - 1, j], completion_time[:, i, j - 1])
    return completion_time[:, -1, -1]


class FSP(object):
    def __init__(self, name: str):
        self.name = 'FSP-%s' % name
        self.processing_time_matrix = parse_fsp_benchmark(name)
        self.permutation_size = self.processing_time_matrix.size()[0]
        self.maximize = False
        self.n_cores = 1

    def __call__(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return evaluate_fsp_time(x, processing_time_matrix=self.processing_time_matrix)


if __name__ == '__main__':
    #   car1   11
    #   car2   13
    #   car3   12
    #   car4   14
    #   car5   10
    #   car6    8
    #   car7    7
    #   car8    8
    #   hel1  100
    #   hel2   20
    #  reC01   20
    #  reC03   20
    #  reC05   20
    #  reC07   20
    #  reC09   20
    #  reC11   20
    #  reC13   20
    #  reC15   20
    #  reC17   20
    #  reC19   30
    #  reC21   30
    #  reC23   30
    #  reC25   30
    #  reC27   30
    #  reC29   30
    #  reC31   50
    #  reC33   50
    #  reC35   50
    #  reC37   75
    #  reC39   75
    #  reC41   75

    _name = 'reC41'
    _proc_time_mat = parse_fsp_benchmark(_name)
    _n_jobs, _n_machine = _proc_time_mat.size()
    _n_p = 1000
    _p = torch.stack([torch.randperm(_n_jobs) for _ in range(_n_p)], dim=0)
    evaluate_fsp_time(_p, _proc_time_mat)
    print('Done %d jobs %d machines' % (_n_jobs, _n_machine))
