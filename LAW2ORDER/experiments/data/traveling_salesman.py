import numpy as np

import torch

from LAW2ORDER.experiments.config_file_path import tsp_benchmark_filepath


def parse_tsp_benchmark(name: str):
    with open(tsp_benchmark_filepath(name), 'rt') as f:
        tsp_data = f.readlines()
    line_no = 0
    data_info = dict()
    while 'EDGE_WEIGHT_TYPE' not in tsp_data[line_no]:
        line_data = tsp_data[line_no].split(':')
        data_info[line_data[0].strip()] = line_data[1].strip()
        line_no += 1
    line_data = tsp_data[line_no].split(':')
    data_info[line_data[0].strip()] = line_data[1].strip()
    line_no += 1
    assert data_info['TYPE'] == 'TSP'
    dimension = int(data_info['DIMENSION'])
    edge_weight_type = data_info['EDGE_WEIGHT_TYPE']
    if edge_weight_type == 'EXPLICIT':
        while 'EDGE_WEIGHT_FORMAT' not in tsp_data[line_no]:
            line_no += 1
        line_data = tsp_data[line_no].split(':')
        edge_weight_format = line_data[1].strip()
        line_no += 1
        while 'EDGE_WEIGHT_SECTION' not in tsp_data[line_no]:
            line_no += 1
        line_no += 1
        if edge_weight_format == 'UPPER_ROW':
            weight_matrix = np.zeros((dimension, dimension))
            for i in range(dimension - 1):
                weight_matrix[i, i + 1:] = [float(elm) for elm in tsp_data[line_no + i].strip().split()]
            weight_matrix = weight_matrix + weight_matrix.T
        elif edge_weight_format == 'LOWER_DIAG_ROW':
            numbers_list = []
            while 'EOF' not in tsp_data[line_no]:
                numbers_list += [float(elm) for elm in tsp_data[line_no].split()]
                line_no += 1
            weight_matrix = np.zeros((dimension, dimension))
            ind = 0
            for i in range(dimension):
                weight_matrix[i, :i + 1] = numbers_list[ind:ind + i + 1]
                ind += i + 1
            weight_matrix = weight_matrix + weight_matrix.T
        elif edge_weight_format == 'FULL_MATRIX':
            weight_matrix = np.zeros((dimension, dimension))
            for i in range(dimension):
                weight_matrix[i] = [float(elm) for elm in tsp_data[line_no + i].strip().split()]
        else:
            raise NotImplementedError
        for i in range(dimension):
            weight_matrix[i, i] = np.inf
        return torch.from_numpy(weight_matrix.astype(np.float32))
    elif edge_weight_type in ['EUC_2D', 'ATT', 'GEO', 'CEIL_2D']:
        while 'NODE_COORD_SECTION' not in tsp_data[line_no]:
            line_no += 1
        line_no += 1
        coordinates = torch.zeros((dimension, 2))
        for i in range(dimension):
            coordinates[i] = torch.FloatTensor([float(elm) for elm in tsp_data[line_no + i].strip().split()][1:])
        if edge_weight_type == 'EUC_2D':
            weight_matrix = torch.sum(
                (coordinates.view(dimension, 1, 2) - coordinates.view(1, dimension, 2)) ** 2, dim=2) ** 0.5
        elif edge_weight_type == 'ATT':
            weight_matrix = torch.sum(
                (coordinates.view(dimension, 1, 2) - coordinates.view(1, dimension, 2)) ** 2 / 10.0, dim=2) ** 0.5
            weight_matrix += torch.round(weight_matrix) < weight_matrix
        elif edge_weight_type == 'GEO':
            deg = torch.round(coordinates)
            lat_lon = np.pi * (deg + 0.5 * (coordinates - deg) / 3.0) / 180.0
            rrr = 6373.388
            q1 = torch.cos(lat_lon[:, 1].view(-1, 1) - lat_lon[:, 1].view(1, -1))
            q2 = torch.cos(lat_lon[:, 0].view(-1, 1) - lat_lon[:, 0].view(1, -1))
            q3 = torch.cos(lat_lon[:, 0].view(-1, 1) + lat_lon[:, 0].view(1, -1))
            weight_matrix = (rrr * torch.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0).long().float()
        elif edge_weight_type == 'CEIL_2D':
            weight_matrix = torch.sum(
                (coordinates.view(dimension, 1, 2) - coordinates.view(1, dimension, 2)) ** 2, dim=2) ** 0.5
            weight_matrix = torch.ceil(weight_matrix)
        for i in range(dimension):
            weight_matrix[i, i] = np.inf
        return weight_matrix
    else:
        raise ValueError


def evaluate_tsp_cost(permutation: torch.Tensor, distance_matrix: torch.Tensor):
    if permutation.ndim == 1:
        permutation = permutation.view(1, -1)
    begin = permutation
    end = torch.cat([permutation[:, 1:], permutation[:, :1]], dim=1)
    length = torch.sum(distance_matrix[begin, end], dim=1)
    return length


class TSP(object):
    def __init__(self, name: str):
        self.name = 'TSP-%s' % name
        self.distance_matrix = parse_tsp_benchmark(name)
        self.permutation_size = self.distance_matrix.size()[0]
        self.maximize = False
        self.n_cores = 1

    def __call__(self, x):
        return evaluate_tsp_cost(x, distance_matrix=self.distance_matrix)


if __name__ == '__main__':
    #      att48   48
    #     bayg29   29
    #     bays29   29
    #   berlin52   52
    #   brazil58   58
    #    burma14   14
    #      eil51   51
    #      eil76   76
    #      fri26   26
    #       gr17   17
    #       gr21   21
    #       gr24   24
    #       gr48   48
    #       gr96   96
    #       hk48   48
    #    kroA100  100
    #    kroB100  100
    #    kroC100  100
    #    kroD100  100
    #    kroE100  100
    #       pr76   76
    #      rat99   99
    #      rd100  100
    #       st70   70
    #    swiss42   42
    #  ulysses16   16
    #  ulysses22   22

    _wgt_mat = parse_tsp_benchmark('bayg29')
    _p_size = _wgt_mat.size()[0]
    _opt = torch.LongTensor(
        [1, 28, 6, 12, 9, 26, 3, 29, 5, 21, 2, 20, 10, 4, 15, 18, 14, 17, 22, 11, 19, 25, 7, 23, 8, 27, 16, 13, 24]) - 1
    print(evaluate_tsp_cost(_opt, _wgt_mat))
    _value_list = []
    for _ in range(10000):
        _value_list.append(evaluate_tsp_cost(torch.randperm(_p_size), _wgt_mat).item())
    print(min(_value_list))
