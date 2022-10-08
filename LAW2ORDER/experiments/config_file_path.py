from typing import List, Tuple

import os
import socket
from datetime import datetime


HOSTNAME = socket.gethostname()
if HOSTNAME == 'DTA160000':
    DATA_DIR = '/home/coh1/Data'
    EXP_DIR = '/home/coh1/Experiments/LAW2ORDER'
    REPO_DIR = '/home/coh1/git_repositories/LAW2ORDER'
    BATCHCOMBO_EXP_DIR = '/home/coh1/Experiments/BatchCOMBO'
elif HOSTNAME == 'wan-bluechip-BUSINESSline-Workstat':
    DATA_DIR = '/home/changyong/Data'
    EXP_DIR = '/home/changyong/Experiments/LAW2ORDER'
    REPO_DIR = '/home/changyong/git_repositories/LAW2ORDER'
    BATCHCOMBO_EXP_DIR = '/home/changyong/Experiments/BatchCOMBO'
elif HOSTNAME[:4] == 'node' or HOSTNAME[:2] == 'fs':  # DAS5
    DATA_DIR = '/var/scratch/coh/Data'
    EXP_DIR = '/var/scratch/coh/Experiments/LAW2ORDER'
    REPO_DIR = '/var/scratch/coh/git_repositories/LAW2ORDER'
    BATCHCOMBO_EXP_DIR = '/var/scratch/coh/Experiments/BatchCOMBO'
elif 'lisa.surfsara.nl' in HOSTNAME or 'lisa.surfsara.nl' in HOSTNAME:
    DATA_DIR = '/home/cyoh/Data'
    EXP_DIR = '/home/cyoh/Experiments/LAW2ORDER'
    REPO_DIR = '/home/cyoh/git_repositories/LAW2ORDER'
    BATCHCOMBO_EXP_DIR = '/home/cyoh/Experiments/BatchCOMBO'


if not os.path.exists(EXP_DIR):
    os.makedirs(EXP_DIR)

if not os.path.exists(BATCHCOMBO_EXP_DIR):
    os.makedirs(BATCHCOMBO_EXP_DIR)


def floorplan_benchmark_filepath(name: str) -> str:
    return os.path.join(DATA_DIR, 'floorplanning', '%s' % name)


def qap_benchmark_filepath(name: str) -> str:
    return os.path.join(DATA_DIR, 'quadratic_assignment_problems/%s.dat' % name)


def fsp_benchmark_filepath() -> str:
    return os.path.join(DATA_DIR, 'flowshop_scheduling_problems/flowshop1.txt')


def tsp_benchmark_filepath(name: str) -> str:
    return os.path.join(DATA_DIR, 'traveling_salesman_problems/%s.tsp' % name)


def atsp_benchmark_filepath(name: str) -> str:
    return os.path.join(DATA_DIR, 'asymmetric_traveling_salesman_problems/%s.atsp' % name)


def wtp_benchmark_filepath(name: str) -> str:
    return os.path.join(DATA_DIR, 'weighted_tardiness_problems/%s.dat' % name)


def regression_data_filepath(name: str) -> str:
    if not os.path.exists(os.path.join(DATA_DIR, 'permutation_regression')):
        os.makedirs(os.path.join(DATA_DIR, 'permutation_regression'))
    return os.path.join(DATA_DIR, 'permutation_regression', name)


def regression_result_filepath(name: str) -> str:
    if not os.path.exists(os.path.join(EXP_DIR, 'permutation_regression')):
        os.makedirs(os.path.join(EXP_DIR, 'permutation_regression'))
    return os.path.join(EXP_DIR, 'permutation_regression', name)


def variational_optimization_result_filepath() -> str:
    return os.path.join(EXP_DIR, 'variational_optimization')


def generate_time_tag():
    return datetime.now().strftime('%y%m%d-%H%M%S-%f')


def generate_bo_exp_dirname(
        gp_model, acquisition_type, evaluator, batch_size: int, seed: int, exp_dir: str, with_time_tag: bool):
    kernel_name = gp_model.kernel_name_str()
    dirname = '_'.join([evaluator.name, acquisition_type, 'B%02d' % batch_size, kernel_name, 'R%02d' % seed])
    if with_time_tag:
        dirname = '_'.join([dirname, generate_time_tag()])
    if not os.path.exists(os.path.join(exp_dir, dirname)):
        os.makedirs(os.path.join(exp_dir, dirname))
    return dirname
