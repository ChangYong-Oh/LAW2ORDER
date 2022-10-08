from typing import Callable, Tuple, List, Dict, Optional

import os
import socket
import argparse
import time
import pickle
import dill
import subprocess
import warnings
import shlex
import fcntl
from enum import Enum
from pathos import multiprocessing
import multiprocess.context as ctx

import numpy as np

import torch
from torch import Tensor

from gpytorch.kernels import ScaleKernel

from LAW2ORDER.gpytorch_bop.kernels.permutation_kernels import PositionKernel
from LAW2ORDER.surrogate.gp_models import ExactGPRegression
from LAW2ORDER.experiments.data import DAGNML
from LAW2ORDER.experiments.config_file_path import EXP_DIR, generate_bo_exp_dirname, generate_time_tag
from LAW2ORDER.experiments.tasks.bo_permutation import suggest_batch, save_bo, print_bo, load_bo

DAG_EXP_DIR = os.path.join(os.path.split(EXP_DIR)[0], os.path.split(EXP_DIR)[1] + 'DAG')
HOSTNAME = socket.gethostname()
TMP_EVAL_DIR = 'tmp_eval_files'
FILENAME_ZFILL_SIZE = 4
ctx._force_start_method('spawn')
N_MODEL_INITS = 10
ACQ_OPTIM_N_INITS = 30


class DirStatus(Enum):
    Suggesting = "Suggesting"
    Acquired = "Acquired"
    Evaluating = "Evaluating"
    Evaluated = "Evaluated"
    Completed = "Completed"


class EvalFileStatus(Enum):
    Waiting = "Waiting"
    Evaluating = "Evaluating"
    Completed = "Completed"


def n_my_jobs(user: str = 'coh'):
    if HOSTNAME in ['fs0', 'fs1', 'fs2', 'fs3', 'fs4']:
        process = subprocess.Popen(['squeue', '-u', user], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = process.communicate()
        stdout = stdout.decode('utf-8')
        return len(stdout.split('\n')) - 2
    else:
        return None


def repair_broken_exp(dir_name: str):
    bo_data = load_bo(dir_name=dir_name, exp_dir=DAG_EXP_DIR)
    batch_size = bo_data['batch_size']
    data_x = bo_data['data_x']
    data_y = bo_data['data_y']
    evaluator = bo_data['evaluator']
    n_data_x = data_x.size()[0]
    n_data_y = data_y.size()[0]
    assert n_data_x - n_data_y <= batch_size
    existing_eval_file_status_list = []
    if n_data_y < n_data_x:
        for idx in range(n_data_y, n_data_x):
            filename = most_updated_eval_file_name(dir_name=dir_name, torch_idx=idx)
            if filename is None:
                filename_waiting = \
                    generate_eval_file_name(dir_name=dir_name, torch_idx=idx, status=EvalFileStatus.Waiting.value)
                eval_data = {'Status': EvalFileStatus.Waiting.value, 'evaluator': evaluator, 'idx': idx,
                             'data_x': data_x[idx].view(1, -1), 'data_y': None, 'eval_time': None}
                write_pickle(file_name=filename_waiting, data=eval_data)
            else:
                with open(filename, 'rb') as f:
                    eval_data = pickle.load(f)
                filename_evaluating = \
                    generate_eval_file_name(dir_name=dir_name, torch_idx=idx, status=EvalFileStatus.Evaluating.value)
                filename_completed = \
                    generate_eval_file_name(dir_name=dir_name, torch_idx=idx, status=EvalFileStatus.Completed.value)
                if eval_data['Status'] == EvalFileStatus.Evaluating.value:
                    print('%s has Evaluating status' % filename)
                    eval_data['Status'] = EvalFileStatus.Waiting.value
                    eval_data['data_y'] = None
                    eval_data['eval_time'] = None
                    os.remove(filename_evaluating)
                elif eval_data['Status'] == EvalFileStatus.Completed.value:
                    if eval_data['data_y'] is None or eval_data['eval_time'] is None:
                        print('%s has Completed status with incomplete values' % filename)
                        eval_data['Status'] = EvalFileStatus.Waiting.value
                        eval_data['data_y'] = None
                        eval_data['eval_time'] = None
                        os.remove(filename_completed)
                        if os.path.exists(filename_evaluating):
                            os.remove(filename_evaluating)
                with open(filename, 'wb') as f:
                    pickle.dump(eval_data, f)
            existing_eval_file_status_list.append(most_updated_eval_file_name(dir_name=dir_name, torch_idx=idx))
    else:
        for idx in range(n_data_x, n_data_x + batch_size):
            filename_waiting = \
                generate_eval_file_name(dir_name=dir_name, torch_idx=idx, status=EvalFileStatus.Waiting.value)
            if os.path.exists(filename_waiting):
                os.remove(filename_waiting)
            filename_evaluating = \
                generate_eval_file_name(dir_name=dir_name, torch_idx=idx, status=EvalFileStatus.Evaluating.value)
            if os.path.exists(filename_evaluating):
                os.remove(filename_evaluating)
            filename_completed = \
                generate_eval_file_name(dir_name=dir_name, torch_idx=idx, status=EvalFileStatus.Completed.value)
            if os.path.exists(filename_completed):
                os.remove(filename_completed)
            existing_eval_file_status_list.append(most_updated_eval_file_name(dir_name=dir_name, torch_idx=idx))
    print(dir_name)
    print('%d inputs and %d outputs' % (n_data_x, n_data_y))
    print('Individual evaluation files with following numbering exist')
    for file_name in sorted(existing_eval_file_status_list):
        print(os.path.split(file_name)[1])


def generate_bash_command(cmd_type: str, dir_name: str, file_num: Optional[int] = None) -> str:
    assert cmd_type in ['suggest', 'evaluate']
    cmd_parsed = []
    if HOSTNAME[:2] == 'fs':  # das5 server
        cmd_parsed += ['srun', '--time=00:10:00' if cmd_type == 'evaluate' else '--time=02:00:00']
        if HOSTNAME in ['fs0', 'fs2', 'fs4']:
            cmd_parsed += ['-C', 'cpunode']
    elif HOSTNAME == 'DTA160000':
        pass
    else:
        raise NotImplementedError
    cmd_parsed += ['python', '-W', 'ignore', __file__, '--run_type', cmd_type, '--dir_name', dir_name, ]
    if cmd_type == 'evaluate':
        cmd_parsed += ['--i', str(file_num)]
    return ' '.join(cmd_parsed)


def initialize_bo_bash_command(dir_name: str):
    cmd_parsed = []
    if HOSTNAME[:2] == 'fs':  # das5 server
        cmd_parsed += ['srun', '--time=00:15:00']
        if HOSTNAME in ['fs0', 'fs2', 'fs4']:
            cmd_parsed += ['-C', 'cpunode']
    elif HOSTNAME == 'DTA160000':
        pass
    else:
        raise NotImplementedError
    cmd_parsed += ['python', '-W', 'ignore', __file__, '--run_type', 'initbo', '--dir_name', dir_name, ]
    return ' '.join(cmd_parsed)


def generate_eval_file_name(dir_name, torch_idx, status: str):
    if not os.path.exists(os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR)):
        os.makedirs(os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR))
    file_num = torch_idx + 1
    file_name = os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR,
                             '%s_%s_%s.pkl' % (dir_name, str(file_num).zfill(FILENAME_ZFILL_SIZE), status))
    return file_name


def most_updated_eval_file_name(dir_name, torch_idx):
    file_num = torch_idx + 1
    filename_completed = \
        os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR, '%s_%s_%s.pkl'
                     % (dir_name, str(file_num).zfill(FILENAME_ZFILL_SIZE), EvalFileStatus.Completed.value))
    if os.path.exists(filename_completed):
        return filename_completed
    filename_evaluating = \
        os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR, '%s_%s_%s.pkl'
                     % (dir_name, str(file_num).zfill(FILENAME_ZFILL_SIZE), EvalFileStatus.Evaluating.value))
    if os.path.exists(filename_evaluating):
        return filename_evaluating
    filename_waiting = \
        os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR, '%s_%s_%s.pkl'
                     % (dir_name, str(file_num).zfill(FILENAME_ZFILL_SIZE), EvalFileStatus.Waiting.value))
    if os.path.exists(filename_waiting):
        return filename_waiting
    return None


def write_pickle(file_name: str, data):
    with open(file_name, 'wb') as f:
        pickle.dump(obj=data, file=f)


def read_pickle(file_name: str):
    with open(file_name, 'rb') as f:
        return pickle.load(f)


def suggest_postprocess(dir_name: str):
    """
    This should be called for a job that was previously 'Suggesting' status and that has been terminated properly.
    :param dir_name:
    :return:
    """
    if not os.path.exists(os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR)):
        os.makedirs(os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR))
    data = load_bo(dir_name=dir_name, exp_dir=DAG_EXP_DIR)
    batch_size = data['batch_size']
    evaluator = data['evaluator']
    data_x = data['data_x']
    n_data_x = data_x.size()[0]
    n_data_y = data['data_y'].size()[0]
    assert n_data_x - n_data_y <= batch_size
    for idx in range(n_data_y, n_data_x):
        filename = generate_eval_file_name(dir_name=dir_name, torch_idx=idx, status=EvalFileStatus.Waiting.value)
        eval_data = {'Status': EvalFileStatus.Waiting.value, 'evaluator': evaluator, 'idx': idx,
                     'data_x': data_x[idx].view(1, -1), 'data_y': None, 'eval_time': None}
        write_pickle(file_name=filename, data=eval_data)


def get_individual_eval_status(dir_name: str):
    data = load_bo(dir_name=dir_name, exp_dir=DAG_EXP_DIR)
    batch_size = data['batch_size']
    data_x = data['data_x']
    data_y = data['data_y']
    n_data_x = data_x.size()[0]
    n_data_y = data_y.size()[0]
    assert n_data_x - n_data_y <= batch_size
    individual_eval_status_dict = dict()
    for idx in range(n_data_y, n_data_x):
        filename = most_updated_eval_file_name(dir_name=dir_name, torch_idx=idx)
        status = None
        if EvalFileStatus.Waiting.value in filename:
            status = EvalFileStatus.Waiting.value
        elif EvalFileStatus.Evaluating.value in filename:
            status = EvalFileStatus.Evaluating.value
        elif EvalFileStatus.Completed.value in filename:
            status = EvalFileStatus.Completed.value
        individual_eval_status_dict[idx] = status
    return individual_eval_status_dict


def get_tardiest_dir_name(eval_status_dict: Dict):
    """

    :param eval_status_dict: {dir_name: {idx: Status}} here the idx tells the process of dir_name
    :return:
    """
    n_eval_dict = dict()
    for dir_name, eval_status in eval_status_dict.items():
        n_eval_dict[dir_name] = max(list(eval_status.keys()))
    return min(n_eval_dict, key=n_eval_dict.get)


class JobManager(object):
    def __init__(self, max_eval: int, n_parallel: int, dir_name_list: Optional[List[str]] = None):
        """
        dir_name is the experiment id
        :param n_parallel:
        :param dir_name_list:
        """
        self._dir_name_list = dir_name_list
        self._max_eval = max_eval
        self._n_parallel = n_parallel
        self._dir_status_dict = dict()
        for dir_name in self._dir_name_list:
            data = load_bo(dir_name=dir_name, exp_dir=DAG_EXP_DIR)
            batch_size = data['batch_size']
            n_data_x = data['data_x'].size()[0]
            n_data_y = data['data_y'].size()[0]
            assert n_data_x - n_data_y in [0, batch_size]
            if n_data_y == max_eval:
                self._dir_status_dict[dir_name] = DirStatus.Completed.value
            else:  # Having DirStatus.Suggesting initially is impossible
                if n_data_x - n_data_y == 0:
                    self._dir_status_dict[dir_name] = DirStatus.Evaluated.value
                else:  # n_data_x - n_data_y == batch_size
                    eval_data_status = get_individual_eval_status(dir_name=dir_name)
                    assert list(eval_data_status.values()).count(EvalFileStatus.Evaluating.value) == 0
                    self._dir_status_dict[dir_name] = self._evaluate_postprocess(dir_name)
        print('Job Manager has been initialized by setting directory status')

    def _evaluate_postprocess(self, dir_name: str):
        eval_status = get_individual_eval_status(dir_name=dir_name)
        bo_data = load_bo(dir_name=dir_name, exp_dir=DAG_EXP_DIR)
        batch_size = bo_data['batch_size']
        if list(eval_status.values()).count(EvalFileStatus.Completed.value) == batch_size:
            data_x = bo_data['data_x']
            data_y = bo_data['data_y']
            eval_time = bo_data['eval_time']
            time_stamp = bo_data['time_stamp']
            data_y = torch.cat([data_y, data_y.new_zeros(batch_size)], dim=0)
            eval_time = torch.cat([eval_time, eval_time.new_zeros(batch_size)], dim=0)
            time_stamp: List = time_stamp + [time.time() for _ in range(batch_size)]
            for idx, status in eval_status.items():
                assert status == EvalFileStatus.Completed.value
                eval_data = read_pickle(generate_eval_file_name(dir_name=dir_name, torch_idx=idx, status=status))
                assert torch.all(data_x[idx] == eval_data['data_x'].view(-1))
                data_y[idx] = eval_data['data_y']
                eval_time[idx] = eval_data['eval_time']
            bo_data['data_y'] = data_y
            bo_data['eval_time'] = eval_time
            bo_data['time_stamp'] = time_stamp
            save_bo(bo_data=bo_data, dir_name=dir_name, exp_dir=DAG_EXP_DIR)
            return DirStatus.Evaluated.value if data_y.size()[0] < self._max_eval else DirStatus.Completed.value
        else:
            return DirStatus.Evaluating.value

    def _new_job_bash_command(self) -> Tuple[str, str, str, int]:
        # exp in the middle of evaluations of multiple points
        exp_evaluating = [elm for elm in self._dir_name_list
                          if self._dir_status_dict[elm] == DirStatus.Evaluating.value]
        # exp which needs to start batch evaluation
        exp_to_evaluate = [elm for elm in self._dir_name_list
                           if self._dir_status_dict[elm] == DirStatus.Acquired.value]
        # exp which needs to start a new round from surrogate fitting
        exp_to_suggest = [elm for elm in self._dir_name_list
                          if self._dir_status_dict[elm] == DirStatus.Evaluated.value]
        # below if elif is in order to give the priority in the order
        # print('%4d in Evaluating Status' % len(exp_evaluating))
        # print('%4d in Acquired Status waiting to move to Evaluating Status' % len(exp_to_evaluate))
        # print('%4d in Evaluated Status waiting to move to Suggesting Status' % len(exp_to_suggest))

        if len(exp_to_suggest) > 0:
            exp_to_suggest_eval_num_dict = {
                dir_name: load_bo(dir_name=dir_name, exp_dir=DAG_EXP_DIR)['data_y'].size()[0]
                for dir_name in exp_to_suggest}
            tardiest_dir_name = min(exp_to_suggest_eval_num_dict, key=exp_to_suggest_eval_num_dict.get)
            bash_cmd = generate_bash_command(cmd_type='suggest', dir_name=tardiest_dir_name)
            print('Following command is going to be submitted\n%s' % bash_cmd)
            return bash_cmd, tardiest_dir_name, 'suggest', 0

        if len(exp_to_evaluate) > 0:
            # To choose the exp with the smallest number of BO rounds
            exp_to_evaluate_eval_status_dict = {dir_name: get_individual_eval_status(dir_name=dir_name)
                                                for dir_name in exp_to_evaluate}
            tardiest_dir_name = get_tardiest_dir_name(exp_to_evaluate_eval_status_dict)
            individual_eval_status = exp_to_evaluate_eval_status_dict[tardiest_dir_name]
            torch_idx = min(list(individual_eval_status.keys()))
            file_num = torch_idx + 1
            bash_cmd = generate_bash_command(cmd_type='evaluate', dir_name=tardiest_dir_name, file_num=file_num)
            print('Following command is going to be submitted\n%s' % bash_cmd)
            return bash_cmd, tardiest_dir_name, 'evaluate', torch_idx

        if len(exp_evaluating) > 0:
            # To choose the exp with the smallest number of BO rounds
            exp_evaluating_eval_status_dict = {dir_name: get_individual_eval_status(dir_name=dir_name)
                                               for dir_name in exp_evaluating}
            tardiest_dir_name = get_tardiest_dir_name(exp_evaluating_eval_status_dict)
            individual_eval_status = exp_evaluating_eval_status_dict[tardiest_dir_name]
            waiting_idxs = [idx for idx, eval_status in individual_eval_status.items()
                            if eval_status == EvalFileStatus.Waiting.value]
            if len(waiting_idxs) == 0:  # no waiting --> all evaluating or completed
                self._dir_status_dict[tardiest_dir_name] = self._evaluate_postprocess(dir_name=tardiest_dir_name)
            else:
                torch_idx = min(waiting_idxs)
                bash_cmd = generate_bash_command(cmd_type='evaluate', dir_name=tardiest_dir_name,
                                                 file_num=torch_idx + 1)
                print('Following command is going to be submitted\n%s' % bash_cmd)
                return bash_cmd, tardiest_dir_name, 'evaluate', torch_idx

        return '', '', '', 0

    def run_jobs(self):
        # Starting JobManager
        process_dict = dict()
        for _ in range(self._n_parallel):
            bash_cmd, dir_name_to_run, job_type, torch_idx = self._new_job_bash_command()
            if len(bash_cmd) > 0:
                process_dict[dir_name_to_run] = subprocess.Popen(shlex.split(bash_cmd), stdout=subprocess.DEVNULL)
                if job_type == 'suggest':
                    assert self._dir_status_dict[dir_name_to_run] == DirStatus.Evaluated.value
                    self._dir_status_dict[dir_name_to_run] = DirStatus.Suggesting.value
                elif job_type == 'evaluate':
                    assert self._dir_status_dict[dir_name_to_run] \
                           in [DirStatus.Acquired.value, DirStatus.Evaluating.value]
                    self._dir_status_dict[dir_name_to_run] = DirStatus.Evaluating.value
                    eval_status_from_waiting_to_evaluating(dir_name=dir_name_to_run, torch_idx=torch_idx)
                else:
                    raise ValueError('Inappropriate job type : %s' % job_type)
        process_prev_state = {dir_name: process.poll() for dir_name, process in process_dict.items()}
        print('The first %d jobs has been submitted.' % self._n_parallel)

        # Continuing jobs
        while not np.all([dir_status == DirStatus.Completed.value
                          for dir_name, dir_status in self._dir_status_dict.items()]):
            n_jobs = n_my_jobs()
            while n_jobs >= self._n_parallel:
                time.sleep(10)
                n_jobs = n_my_jobs()
            # even with for loop if there is only one is processed and loop is escaped
            completed_process_dir_name_list = [dir_name for dir_name, process in process_dict.items()
                                               if process.poll() is not None]
            for dir_name in completed_process_dir_name_list:
                completion_state = process_dict[dir_name].poll()
                if completion_state == 0:  # Processing for the properly terminated job
                    # Meaning that below two cases are the results of running run_dag_bo
                    if self._dir_status_dict[dir_name] == DirStatus.Suggesting.value:  # not-completed suggestion
                        self._dir_status_dict[dir_name] = DirStatus.Acquired.value
                    elif self._dir_status_dict[dir_name] == DirStatus.Evaluating.value:
                        self._dir_status_dict[dir_name] = self._evaluate_postprocess(dir_name)
                    elif self._dir_status_dict[dir_name] in [DirStatus.Evaluated.value, DirStatus.Acquired.value]:
                        pass  # Do nothing until it is picked in _new_job_bash_command
                    elif self._dir_status_dict[dir_name] == DirStatus.Completed.value:
                        pass  # Do nothing when it is completed
                else:  # Processing for the improperly terminated job
                    print("%s was NOT properly processed having began with %s"
                          % (dir_name, self._dir_status_dict[dir_name]))
                del process_dict[dir_name]
            bash_cmd, dir_name_to_run, job_type, torch_idx = self._new_job_bash_command()
            if len(bash_cmd) > 0:
                process_dict[dir_name_to_run] = subprocess.Popen(shlex.split(bash_cmd), stdout=subprocess.DEVNULL)
                if job_type == 'suggest':
                    self._dir_status_dict[dir_name_to_run] = DirStatus.Suggesting.value
                elif job_type == 'evaluate':
                    self._dir_status_dict[dir_name_to_run] = DirStatus.Evaluating.value
                    eval_status_from_waiting_to_evaluating(dir_name=dir_name_to_run, torch_idx=torch_idx)
                else:
                    raise ValueError('Inappropriate job type : %s' % job_type)

        while [process.poll() for process in process_dict.values()].count(None) > 0:
            time.sleep(10)
        print('\n')
        print('\n')
        print('\n')


def run_dag_suggest(gp_model: ExactGPRegression, acquisition_type: str, evaluator: DAGNML, batch_size: int,
                    data_x: torch.Tensor, data_y: torch.Tensor, batch_idx: torch.Tensor,
                    bo_time: torch.Tensor, eval_time: torch.Tensor, time_stamp: List, **kwargs):
    n_data_x = data_x.size()[0]
    n_data_y = data_y.size()[0]
    assert n_data_x == bo_time.size()[0] == batch_idx.size()[0]
    assert n_data_y == eval_time.size()[0] == len(time_stamp)
    assert n_data_x == n_data_y

    bo_start_time = time.time()
    next_x_batch, info_str_suggest = suggest_batch(
        gp_model=gp_model, acquisition_type=acquisition_type, batch_size=batch_size,
        data_x=data_x, data_y=data_y, maximize=evaluator.maximize)
    bo_time = torch.cat([bo_time, torch.ones(batch_size) * (time.time() - bo_start_time)], dim=0)
    data_x = torch.cat([data_x, next_x_batch.view(batch_size, -1).long()], dim=0)
    batch_idx = torch.cat([batch_idx, torch.ones(batch_size).long() * (batch_idx[-1] + 1)], dim=0)
    bo_data = {'gp_model': gp_model, 'acquisition_type': acquisition_type, 'batch_size': batch_size,
               'data_x': data_x, 'data_y': data_y,
               'evaluator': evaluator, 'batch_idx': batch_idx,
               'bo_time': bo_time, 'eval_time': eval_time, 'time_stamp': time_stamp}
    return bo_data, info_str_suggest


def eval_status_from_waiting_to_evaluating(dir_name: str, torch_idx: int):
    filename = most_updated_eval_file_name(dir_name=dir_name, torch_idx=torch_idx)
    eval_data = read_pickle(file_name=filename)
    status = eval_data['Status']
    if status == EvalFileStatus.Evaluating.value:
        print('UserWarning : %s is currently being evaluated' % filename)
        warnings.warn('Status is "Evaluating"')
        return
    elif status == EvalFileStatus.Completed.value:
        print('UserWarning : %s has already been evaluated' % filename)
        warnings.warn('Status is "Completed"')
        return
    eval_data['Status'] = EvalFileStatus.Evaluating.value
    filename_evaluating = generate_eval_file_name(
        dir_name=dir_name, torch_idx=torch_idx, status=EvalFileStatus.Evaluating.value)
    write_pickle(file_name=filename_evaluating, data=eval_data)


def run_dag_evaluator(dir_name: str, torch_idx: int):
    filename = most_updated_eval_file_name(dir_name=dir_name, torch_idx=torch_idx)
    eval_data = read_pickle(file_name=filename)
    filename_completed = generate_eval_file_name(
        dir_name=dir_name, torch_idx=torch_idx, status=EvalFileStatus.Completed.value)
    filename_evaluating = generate_eval_file_name(
        dir_name=dir_name, torch_idx=torch_idx, status=EvalFileStatus.Evaluating.value)
    if eval_data['Status'] == EvalFileStatus.Completed.value:
        if eval_data['data_y'] is None or eval_data['eval_time'] is None:
            eval_data['data_y'] = None
            eval_data['eval_time'] = None
            eval_data['Status'] = EvalFileStatus.Evaluating.value
            os.remove(filename_completed)
            if not os.path.exists(filename_evaluating):
                write_pickle(file_name=filename_evaluating, data=eval_data)
        else:
            return
    elif eval_data['Status'] == EvalFileStatus.Waiting.value:
        eval_data['Status'] = EvalFileStatus.Evaluating.value
        write_pickle(file_name=filename_evaluating, data=eval_data)
    # When called in JobManager then status should be Evaluating
    evaluator = eval_data['evaluator']
    start_time = time.time()
    eval_y = evaluator(eval_data['data_x'].view(1, -1))
    eval_data['eval_time'] = time.time() - start_time
    eval_data['data_y'] = eval_y
    eval_data['Status'] = EvalFileStatus.Completed.value
    write_pickle(file_name=filename_completed, data=eval_data)


def run_dag_bo(run_type: str, dir_name: str, file_num: Optional[int] = None):
    if run_type == 'suggest':
        bo_data, info_str_suggest = run_dag_suggest(**load_bo(dir_name=dir_name, exp_dir=DAG_EXP_DIR))
        save_bo(bo_data=bo_data, dir_name=dir_name, exp_dir=DAG_EXP_DIR)
        suggest_postprocess(dir_name=dir_name)
    elif run_type == 'evaluate':
        run_dag_evaluator(dir_name=dir_name, torch_idx=file_num - 1)
    else:
        raise NotImplementedError


def set_init_individual_eval_files(evaluator: DAGNML, dir_name: str, seed: int, n_data_x: int):
    seed_list = np.random.RandomState(seed).randint(0, 10000, (n_data_x, ))
    p_size = evaluator.n_nodes
    for idx in range(n_data_x):
        permutation = torch.from_numpy(np.random.RandomState(seed_list[idx]).permutation(p_size))
        filename_waiting = generate_eval_file_name(
            dir_name=dir_name, torch_idx=idx, status=EvalFileStatus.Waiting.value)
        with open(filename_waiting, 'wb') as f:
            pickle.dump({'Status': EvalFileStatus.Waiting.value, 'evaluator': evaluator, 'idx': idx,
                         'data_x': permutation.view(1, -1), 'data_y': None, 'eval_time': None}, f)


def run_multiple_commands_in_parallel(cmd_list: List[str], n_process: int):
    process_list = []
    for i in range(n_process):
        process_list.append(subprocess.Popen(shlex.split(cmd_list[i]), stdout=subprocess.PIPE))
        print('Subprocess with below command has been initiated')
        print(cmd_list[i])
    for i in range(n_process, len(cmd_list)):
        n_jobs = n_my_jobs()
        while n_jobs >= n_process:
            time.sleep(2)
            n_jobs = n_my_jobs()
        process_list.append(subprocess.Popen(shlex.split(cmd_list[i]), stdout=subprocess.PIPE))
        print('Subprocess with below command has been initiated')
        print(cmd_list[i])
    for process in process_list:
        if process.poll() is not None:
            process.wait()
    status = [elm.poll() for elm in process_list]
    while status.count(None) > 0:
        time.sleep(10)
        for i, process in enumerate(process_list):
            if process.poll() is not None:
                process.wait()
                status[i] = process.poll()
    print('\n')
    print('\n')
    print('\n')


def eval_init_individual_eval_files(dir_name_list: List[str], n_process: int):
    cmd_list = []
    for dir_name in dir_name_list:
        for file_name in sorted(os.listdir(os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR))):
            file_num = int(os.path.splitext(file_name)[0].split('_')[-2])
            cmd_list.append(generate_bash_command(cmd_type='evaluate', dir_name=dir_name, file_num=file_num))
    run_multiple_commands_in_parallel(cmd_list=cmd_list, n_process=n_process)


def get_init_eval_file_name(dir_name, torch_idx):
    file_num = torch_idx + 1
    file_name = os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR,
                             '%s_%s.pkl' % (dir_name, str(file_num).zfill(FILENAME_ZFILL_SIZE)))
    return file_name


def initialize_bo_from_eval_data(dir_name: str, batch_size: int, acquisition_type: str):
    eval_data_dict = dict()
    for idx in range(batch_size):
        with open(get_init_eval_file_name(dir_name=dir_name, torch_idx=idx), 'rb') as f:
            eval_data_dict[idx] = pickle.load(f)
            assert eval_data_dict[idx]['Status'] == EvalFileStatus.Completed.value
    assert len(eval_data_dict) == max(eval_data_dict.keys()) + 1
    data_x_list = []
    data_y_list = []
    eval_time_list = []
    for idx in eval_data_dict.keys():
        assert eval_data_dict[idx]['Status'] == EvalFileStatus.Completed.value
        data_x_list.append(eval_data_dict[idx]['data_x'].view(1, -1))
        data_y_list.append(eval_data_dict[idx]['data_y'].view(1))
        eval_time_list.append(torch.tensor(eval_data_dict[idx]['eval_time']).view(-1))
    data_x = torch.cat(data_x_list, dim=0)
    data_y = torch.cat(data_y_list, dim=0).view(-1)
    eval_time = torch.cat(eval_time_list, dim=0)

    batch_idx = torch.zeros(DAG_OPTIM_N_INITS).long()
    bo_time = torch.zeros(DAG_OPTIM_N_INITS)
    time_stamp = [time.time() for _ in range(DAG_OPTIM_N_INITS)]

    evaluator = eval_data_dict[0]['evaluator']
    p_size = evaluator.n_nodes
    kernel = ScaleKernel(PositionKernel(permutation_size=p_size))
    gp_model = ExactGPRegression(train_x=data_x, train_y=data_y, kernel=kernel)
    gp_model.init_params()

    bo_data = {'gp_model': gp_model, 'acquisition_type': acquisition_type, 'batch_size': batch_size,
               'data_x': data_x, 'data_y': data_y,
               'evaluator': evaluator, 'batch_idx': batch_idx,
               'bo_time': bo_time, 'eval_time': eval_time, 'time_stamp': time_stamp}
    dir_name_time_tag = '_'.join([dir_name, generate_time_tag()])
    if not os.path.exists(os.path.join(DAG_EXP_DIR, dir_name_time_tag)):
        os.makedirs(os.path.join(DAG_EXP_DIR, dir_name_time_tag))
    save_bo(bo_data=bo_data, dir_name=dir_name_time_tag, exp_dir=DAG_EXP_DIR)
    run_dag_bo(run_type='suggest', dir_name=dir_name_time_tag)


def initialize_bo_in_parallel(dir_name_list: List[str], n_process: int):
    cmd_list = []
    for dir_name in dir_name_list:
        cmd_list.append(initialize_bo_bash_command(dir_name=dir_name))
    run_multiple_commands_in_parallel(cmd_list=cmd_list, n_process=n_process)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Structure Learning')
    parser.add_argument('--run_type', dest='run_type', type=str)
    parser.add_argument('--acquisition', dest='acquisition', type=str)
    parser.add_argument('--data', dest='data', type=str, default=None)
    parser.add_argument('--process', dest='process', type=int, default=None)
    parser.add_argument('--dir_file', dest='dir_file', type=str, default=None)
    parser.add_argument('--dir_name', dest='dir_name', type=str, default=None)
    parser.add_argument('--i', dest='i', type=int, default=None)
    parser.add_argument('--seed', dest='seed', type=int, default=None)

    DAG_OPTIM_N_INITS = 20
    BATCH_SIZE = 20
    MAX_EVAL = 620

    args = parser.parse_args()
    if args.acquisition is not None:
        assert args.acquisition in ['WDPP-posterior-MAX-est', 'WDPP-posterior-MAX-ei', 'qPOINT-est', 'qPOINT-ei']
    assert args.run_type in ['suggest', 'evaluate', 'jobmanager', 'initdata', 'initeval', 'initbo', 'repair']
    if args.run_type in ['suggest', 'evaluate']:
        assert args.dir_name is not None  # args.i can be None only if run_type is suggest
        run_dag_bo(run_type=args.run_type, dir_name=args.dir_name, file_num=args.i)
    elif args.run_type == 'jobmanager':
        assert args.process is not None
        with open(args.dir_file, 'rt') as f_:
            dir_name_list_ = f_.read().strip().split('\n')
        worker = JobManager(max_eval=MAX_EVAL, n_parallel=args.process, dir_name_list=dir_name_list_)
        worker.run_jobs()
    elif args.run_type == 'initdata':
        assert args.data in ['sachs', 'child', 'insurance', 'alarm']
        evaluator_ = DAGNML(args.data)
        p_size_ = evaluator_.n_nodes
        kernel_ = ScaleKernel(PositionKernel(permutation_size=p_size_))
        gp_model_ = ExactGPRegression(train_x=None, train_y=None, kernel=kernel_)
        dir_name_ = dirname = '_'.join([evaluator_.name, 'INITDATA', 'B%02d' % BATCH_SIZE, 'NONE', 'R%02d' % args.seed])
        set_init_individual_eval_files(
            evaluator=evaluator_, dir_name=dir_name_, seed=args.seed, n_data_x=DAG_OPTIM_N_INITS)
    elif args.run_type == 'initeval':
        with open(args.dir_file, 'rt') as f_:
            dir_name_list_ = f_.read().strip().split('\n')
        eval_init_individual_eval_files(dir_name_list=dir_name_list_, n_process=args.process)
    elif args.run_type == 'initbo':
        if args.dir_file is not None:
            assert args.dir_name is None
            with open(args.dir_file, 'rt') as f_:
                dir_name_list_ = f_.read().strip().split('\n')
            initialize_bo_in_parallel(dir_name_list=dir_name_list_, n_process=args.process)
        else:
            assert args.dir_name is not None
            initialize_bo_from_eval_data(
                dir_name=args.dir_name, batch_size=BATCH_SIZE, acquisition_type=args.acquisition)
    elif args.run_type == 'repair':
        if args.dir_file is not None:
            assert args.dir_name is None
            with open(args.dir_file, 'rt') as f_:
                dir_name_list_ = f_.read().strip().split('\n')
            for dir_name_ in dir_name_list_:
                repair_broken_exp(dir_name=dir_name_)
        else:
            assert args.dir_name is not None
            repair_broken_exp(dir_name=args.dir_name)
