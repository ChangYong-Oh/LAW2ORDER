from typing import Dict, List, Optional, Tuple

import os
import sys
import signal
import time
import pickle
import dill
import shlex
import subprocess
import argparse
import socket
import warnings
from enum import Enum

import numpy as np

import torch

from pymoo.algorithms.so_genetic_algorithm import FitnessSurvival
from pymoo.factory import get_selection, get_crossover, get_mutation
from pymoo.model.problem import Problem
from pymoo.model.mating import Mating
from pymoo.model.population import Population
from pymoo.model.duplicate import DefaultDuplicateElimination
from pymoo.model.repair import NoRepair


from LAW2ORDER.experiments.data import DAGNML
from LAW2ORDER.experiments.config_file_path import generate_time_tag
from LAW2ORDER.experiments.tasks.bo_dag import \
    DAG_EXP_DIR, TMP_EVAL_DIR, EvalFileStatus, write_pickle, read_pickle, n_my_jobs, \
    run_multiple_commands_in_parallel


HOSTNAME = socket.gethostname()
GENERATION_ZFILL = 3
FILEID_ZFILL = 3


class DirStatus(Enum):
    Evaluating = "Evaluating"
    Offspring = "Offspring"
    Completed = "Completed"


def generate_ga_exp_dirname(evaluator, batch_size: int, seed: int, exp_dir: str, with_time_tag: bool):
    dirname = '_'.join([evaluator.name, 'GA', 'B%02d' % batch_size, 'GA', 'R%02d' % seed])
    if with_time_tag:
        dirname = '_'.join([dirname, generate_time_tag()])
    if not os.path.exists(os.path.join(exp_dir, dirname)):
        os.makedirs(os.path.join(exp_dir, dirname))
    return dirname


class DAGNMLPymooProblem(Problem):

    def __init__(self, evaluator: DAGNML):
        self.evaluator = evaluator
        assert not evaluator.maximize  # pymoo only minimizes
        super().__init__(n_obj=1, n_var=self.evaluator.n_nodes, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        f = self.evaluator(torch.from_numpy(x).view(1, -1)).item()
        out["F"] = [f]


# by using this, its first 20 is equal to what is used in BBO experiments
def set_init_data_x(evaluator: DAGNML, dir_name: str, seed: int, n_init_data_x: int):
    seed_list = np.random.RandomState(seed).randint(0, 10000, (n_init_data_x, ))
    p_size = evaluator.n_nodes
    for idx in range(n_init_data_x):
        data_x = np.random.RandomState(seed_list[idx]).permutation(p_size)
        permutation = torch.from_numpy(data_x)
        filename_waiting = generate_eval_file_name(
            dir_name=dir_name, generation=1, individual_id=idx, status=EvalFileStatus.Waiting.value)
        with open(filename_waiting, 'wb') as f:
            pickle.dump({'Status': EvalFileStatus.Waiting.value, 'evaluator': evaluator, 'individual_id': idx,
                         'data_x': permutation.view(1, -1), 'data_y': None, 'eval_time': None}, f)


def load_ga(dir_name: str, exp_dir: str):
    filename_list = [elm for elm in os.listdir(os.path.join(exp_dir, dir_name))
                     if os.path.isfile(os.path.join(exp_dir, dir_name, elm))]
    last_filename_postfix = max([os.path.splitext(elm)[0].split('_')[-1] for elm in filename_list])
    last_filename = [elm for elm in filename_list
                     if ('%s.pkl' % last_filename_postfix) == elm[-(4 + len(last_filename_postfix)):]][0]
    with open(os.path.join(exp_dir, dir_name, last_filename), 'rb') as f:
        ga_data = dill.load(f)
    return ga_data


def save_ga(ga_data: Dict, dir_name: str, exp_dir: str):
    generation = ga_data['generation']
    with open(os.path.join(exp_dir, dir_name, '%s_%03d.pkl' % (dir_name, generation)), 'wb') as f:
        dill.dump(ga_data, f)


def generate_eval_file_name(dir_name, generation, individual_id, status: str):
    if not os.path.exists(os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR)):
        os.makedirs(os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR))
    generation_id_str = '%s-%s' % (str(generation).zfill(GENERATION_ZFILL), str(individual_id).zfill(FILEID_ZFILL))
    file_name = os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR,
                             '%s_%s_%s.pkl' % (dir_name, generation_id_str, status))
    return file_name


def most_updated_eval_file_name(dir_name, generation, individual_id):
    generation_id_str = '%s-%s' % (str(generation).zfill(GENERATION_ZFILL), str(individual_id).zfill(FILEID_ZFILL))
    filename_completed = \
        os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR, '%s_%s_%s.pkl'
                     % (dir_name, generation_id_str, EvalFileStatus.Completed.value))
    if os.path.exists(filename_completed):
        return filename_completed
    filename_evaluating = \
        os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR, '%s_%s_%s.pkl'
                     % (dir_name, generation_id_str, EvalFileStatus.Evaluating.value))
    if os.path.exists(filename_evaluating):
        return filename_evaluating
    filename_waiting = \
        os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR, '%s_%s_%s.pkl'
                     % (dir_name, generation_id_str, EvalFileStatus.Waiting.value))
    if os.path.exists(filename_waiting):
        return filename_waiting
    return None


def individual_ids_in_generation(dir_name, generation):
    individual_ids = []
    if not os.path.exists(os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR)):
        os.makedirs(os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR))
    for eval_file in os.listdir(os.path.join(DAG_EXP_DIR, dir_name, TMP_EVAL_DIR)):
        gen_str, id_str = os.path.splitext(eval_file)[0].split('_')[-2].split('-')
        if int(gen_str) == generation:
            individual_ids.append(int(id_str))
    individual_ids = sorted(list(set(individual_ids)))
    return individual_ids


def latest_generation(dir_name):
    return max([int(os.path.splitext(elm)[0].split('_')[-1])
                for elm in os.listdir(os.path.join(DAG_EXP_DIR, dir_name))
                if os.path.isfile(os.path.join(DAG_EXP_DIR, dir_name, elm))])


def ga_get_offsprings(dirname: str):
    ga_data = load_ga(dir_name=dirname, exp_dir=DAG_EXP_DIR)
    pop = ga_data['pop']
    pop.set(F=pop.get('F').reshape((-1, 1)))
    problem = ga_data['problem']
    n_offsprings = ga_data['n_offsprings']
    generation = ga_data['generation']
    evaluator = ga_data['evaluator']
    eliminate_duplicates = DefaultDuplicateElimination()
    repair = NoRepair()
    mating = Mating(
        selection=get_selection("random"), crossover=get_crossover("perm_erx"), mutation=get_mutation("perm_inv"),
        repair=repair, eliminate_duplicates=eliminate_duplicates, n_max_iterations=100)
    off = mating.do(problem, pop, n_offsprings)
    next_gen = generation + 1

    if not os.path.exists(os.path.join(DAG_EXP_DIR, dirname, TMP_EVAL_DIR)):
        os.makedirs(os.path.join(DAG_EXP_DIR, dirname, TMP_EVAL_DIR))

    for individual_id in range(len(off)):
        filename = generate_eval_file_name(dir_name=dirname, generation=next_gen, individual_id=individual_id,
                                           status=EvalFileStatus.Waiting.value)
        data_x = torch.from_numpy(off[individual_id].get('X')).view(1, -1)
        eval_data = {'Status': EvalFileStatus.Waiting.value, 'evaluator': evaluator, 'individual_id': individual_id,
                     'data_x': data_x, 'data_y': None, 'eval_time': None}
        write_pickle(file_name=filename, data=eval_data)
    return


def ga_new_generation(dirname: str):
    next_gen = latest_generation(dirname) + 1
    individual_ids = individual_ids_in_generation(dir_name=dirname, generation=next_gen)
    assert len(individual_ids) == max(individual_ids) + 1
    n_individuals = len(individual_ids)
    off_x = None
    off_y = np.zeros((n_individuals, ))
    eval_time = np.zeros((n_individuals, ))
    for individual_id in individual_ids:
        filename_completed = generate_eval_file_name(dir_name=dirname, generation=next_gen,
                                                     individual_id=individual_id, status=EvalFileStatus.Completed.value)
        assert os.path.exists(filename_completed)
        eval_data = read_pickle(file_name=filename_completed)
        assert eval_data['individual_id'] == individual_id
        if off_x is None:
            off_x = np.zeros((n_individuals, eval_data['data_x'].numel()), dtype=np.int)
        off_x[individual_id] = eval_data['data_x'].numpy()
        off_y[individual_id] = float(eval_data['data_y'])
        eval_time[individual_id] = float(eval_data['eval_time'])
    off = Population.new(X=off_x, F=off_y.reshape((-1, 1)))
    ga_data = load_ga(dir_name=dirname, exp_dir=DAG_EXP_DIR)
    pop = ga_data['pop']
    pop.set(F=pop.get('F').reshape((-1, 1)))
    pop = Population.merge(pop, off)

    pop = FitnessSurvival().do(problem=ga_data['problem'], pop=pop, n_survive=pop.get('X').shape[0])
    ga_data['pop'] = pop
    ga_data['generation'] += 1
    ga_data['eval_time'] = eval_time
    save_ga(ga_data=ga_data, dir_name=dirname, exp_dir=DAG_EXP_DIR)
    return


def most_updated_eval_files_in_next_generation(dir_name):
    next_gen = latest_generation(dir_name) + 1
    individual_ids = individual_ids_in_generation(dir_name=dir_name, generation=next_gen)
    most_updated_eval_files = \
        [most_updated_eval_file_name(dir_name=dir_name, generation=next_gen, individual_id=elm)
         for elm in individual_ids]
    return most_updated_eval_files


def generate_bash_command(cmd_type: str, dir_name: Optional[str] = None, file_name: Optional[str] = None) -> str:
    assert cmd_type in ['suggest', 'evaluate']
    cmd_parsed = []
    if HOSTNAME[:2] == 'fs':  # das5 server
        cmd_parsed += ['srun', '--time=00:05:00']
        if HOSTNAME in ['fs0', 'fs2', 'fs4']:
            cmd_parsed += ['-C', 'cpunode']
    elif HOSTNAME == 'DTA160000':
        pass
    else:
        raise NotImplementedError
    cmd_parsed += ['python', '-W', 'ignore', __file__, '--run_type', cmd_type]
    if cmd_type == 'suggest':
        assert file_name is None
        cmd_parsed += ['--dir_name', dir_name]
    elif cmd_type == 'evaluate':
        assert dir_name is None
        cmd_parsed += ['--file_name', file_name]
    return ' '.join(cmd_parsed)


def eval_status_from_waiting_to_evaluating(filename: str):
    filename_waiting = filename
    dir_part, file_part = os.path.split(filename)
    file_name_part, file_ext_part = os.path.splitext(file_part)
    assert file_name_part.split('_')[-1] == EvalFileStatus.Waiting.value
    filename_evaluating = \
        os.path.join(dir_part,
                     '_'.join(file_name_part.split('_')[:-1] + [EvalFileStatus.Evaluating.value]) + file_ext_part)
    eval_data = read_pickle(file_name=filename_waiting)
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
    write_pickle(file_name=filename_evaluating, data=eval_data)
    return


def kill_process(process):
    assert process.poll() is not None
    stdout, stderr = process.communicate()
    # print(stdout)
    # print(stderr)

    process.kill()
    process.terminate()
    process.wait()
    return


class JobManager(object):
    def __init__(self, max_generation: int, n_parallel: int, dir_name_list: Optional[List[str]] = None):
        """
        dir_name is the experiment id
        :param n_parallel:
        :param dir_name_list:
        """
        self._dir_name_list = dir_name_list
        self._max_generation = max_generation
        self._n_parallel = n_parallel
        self._dir_status_dict = dict()
        for dir_name in self._dir_name_list:
            data = load_ga(dir_name=dir_name, exp_dir=DAG_EXP_DIR)
            generation = data['generation']
            if generation >= max_generation:
                self._dir_status_dict[dir_name] = DirStatus.Completed.value
            else:  # Having DirStatus.Suggesting initially is impossible
                most_updated_eval_files = most_updated_eval_files_in_next_generation(dir_name)
                if len(most_updated_eval_files) == 0:
                    ga_get_offsprings(dirname=dir_name)
                    self._dir_status_dict[dir_name] = DirStatus.Offspring.value
                else:
                    individual_evaluating = \
                        [elm for elm in most_updated_eval_files if EvalFileStatus.Evaluating.value in elm]
                    individual_waiting = \
                        [elm for elm in most_updated_eval_files if EvalFileStatus.Waiting.value in elm]
                    if len(individual_evaluating) + len(individual_waiting) == 0:
                        ga_new_generation(dirname=dir_name)
                        if generation + 1 < MAX_GEN:
                            ga_get_offsprings(dirname=dir_name)
                            self._dir_status_dict[dir_name] = DirStatus.Offspring.value
                        else:
                            self._dir_status_dict[dir_name] = DirStatus.Completed.value
                    else:
                        if len(individual_evaluating) > 0:
                            for indi_eval in individual_evaluating:
                                os.remove(indi_eval)
                        self._dir_status_dict[dir_name] = DirStatus.Evaluating.value
        print('Job Manager has been initialized by setting directory status')

    def _new_job_bash_command(self) -> Tuple[str, str]:
        # exp in the middle of evaluations of multiple points
        exp_evaluating = [elm for elm in self._dir_name_list
                          if self._dir_status_dict[elm] == DirStatus.Evaluating.value]
        # exp which needs to start batch evaluation
        exp_to_evaluate = [elm for elm in self._dir_name_list
                           if self._dir_status_dict[elm] == DirStatus.Offspring.value]

        if len(exp_to_evaluate) > 0:
            exp_to_evaluate_generation = {dir_name: load_ga(dir_name=dir_name, exp_dir=DAG_EXP_DIR)['generation']
                                          for dir_name in exp_to_evaluate}
            tardiest_dir_name = min(exp_to_evaluate_generation, key=exp_to_evaluate_generation.get)
            most_updated_eval_files = most_updated_eval_files_in_next_generation(tardiest_dir_name)
            eval_files_waiting = [elm for elm in most_updated_eval_files if EvalFileStatus.Waiting.value in elm]
            eval_file_to_run = sorted(eval_files_waiting)[0]
            filename_waiting, filename_evaluating, _ = filenames_with_three_status(eval_file_to_run)
            eval_data = read_pickle(filename_waiting)
            eval_data['Status'] = EvalFileStatus.Evaluating.value
            write_pickle(file_name=filename_evaluating, data=eval_data)
            self._dir_status_dict[tardiest_dir_name] = DirStatus.Evaluating.value
            bash_cmd = generate_bash_command(cmd_type='evaluate', file_name=eval_file_to_run)
            print('Following command is going to be submitted\n%s' % bash_cmd)
            return bash_cmd, tardiest_dir_name

        if len(exp_evaluating) > 0:
            exp_evaluating_generation = {dir_name: load_ga(dir_name=dir_name, exp_dir=DAG_EXP_DIR)['generation']
                                         for dir_name in exp_evaluating}
            tardiest_dir_name = min(exp_evaluating_generation, key=exp_evaluating_generation.get)
            tardiest_generation = exp_evaluating_generation[tardiest_dir_name]
            most_updated_eval_files = most_updated_eval_files_in_next_generation(tardiest_dir_name)
            eval_files_waiting = [elm for elm in most_updated_eval_files if EvalFileStatus.Waiting.value in elm]
            eval_files_completed = [elm for elm in most_updated_eval_files if EvalFileStatus.Completed.value in elm]

            if len(eval_files_completed) == len(most_updated_eval_files):  # no waiting --> all evaluating or completed
                ga_new_generation(tardiest_dir_name)
                if tardiest_generation + 1 < MAX_GEN:
                    ga_get_offsprings(dirname=tardiest_dir_name)
                    self._dir_status_dict[tardiest_dir_name] = DirStatus.Offspring.value
                else:
                    self._dir_status_dict[tardiest_dir_name] = DirStatus.Completed.value
            elif len(eval_files_waiting) > 0:
                eval_file_to_run = sorted(eval_files_waiting)[0]
                filename_waiting, filename_evaluating, _ = filenames_with_three_status(eval_file_to_run)
                eval_data = read_pickle(filename_waiting)
                eval_data['Status'] = EvalFileStatus.Evaluating.value
                write_pickle(file_name=filename_evaluating, data=eval_data)
                bash_cmd = generate_bash_command(cmd_type='evaluate', file_name=eval_file_to_run)
                print('Following command is going to be submitted\n%s' % bash_cmd)
                return bash_cmd, tardiest_dir_name

        return '', ''

    def run_jobs(self):
        # Starting JobManager
        process_dict = dict()
        for _ in range(self._n_parallel):
            bash_cmd, dir_to_run = self._new_job_bash_command()
            if len(bash_cmd) > 0:
                process_dict[dir_to_run] = subprocess.Popen(shlex.split(bash_cmd),
                                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('The first %d jobs has been submitted.' % self._n_parallel)

        # Continuing jobs
        while not np.all([dir_status == DirStatus.Completed.value
                          for dir_name, dir_status in self._dir_status_dict.items()]):
            while n_my_jobs() >= self._n_parallel:
                time.sleep(2)
            completed_dir_name_list = []
            for dir_name, process in process_dict.items():
                if process.poll() is not None:
                    kill_process(process_dict[dir_name])
                    completed_dir_name_list.append(dir_name)
            for dirname in completed_dir_name_list:
                del process_dict[dirname]

            bash_cmd, dir_to_run = self._new_job_bash_command()
            if len(bash_cmd) > 0:
                process_dict[dir_to_run] = subprocess.Popen(shlex.split(bash_cmd),
                                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        while [process.poll() for process in process_dict.values()].count(None) > 0:
            time.sleep(10)
        print('\n')
        print('\n')
        print('\n')


def filenames_with_three_status(filename: str):
    dir_part, file_part = os.path.split(filename)
    file_name_part, file_ext_part = os.path.splitext(file_part)
    # gen_id_str = file_name_part.split('_')[-2]
    # generation, individual_id = [int(elm) for elm in gen_id_str.split('-')]
    filename_waiting = \
        os.path.join(dir_part,
                     '_'.join(file_name_part.split('_')[:-1] + [EvalFileStatus.Waiting.value]) + file_ext_part)
    filename_evaluating = \
        os.path.join(dir_part,
                     '_'.join(file_name_part.split('_')[:-1] + [EvalFileStatus.Evaluating.value]) + file_ext_part)
    filename_completed = \
        os.path.join(dir_part,
                     '_'.join(file_name_part.split('_')[:-1] + [EvalFileStatus.Completed.value]) + file_ext_part)
    return filename_waiting, filename_evaluating, filename_completed


def run_ga_evaluator(filename: str):
    filename_waiting, filename_evaluating, filename_completed = filenames_with_three_status(filename)

    if os.path.exists(filename_completed):
        eval_data = read_pickle(file_name=filename_completed)
        if eval_data['data_y'] is None or eval_data['eval_time'] is None:
            eval_data['data_y'] = None
            eval_data['eval_time'] = None
            eval_data['Status'] = EvalFileStatus.Evaluating.value
            os.remove(filename_completed)
            if not os.path.exists(filename_evaluating):
                write_pickle(file_name=filename_evaluating, data=eval_data)
        else:  # Exists a completed file in the correct status
            if not os.path.exists(filename_evaluating):  # doing some fix if there is no Evaluating file
                eval_data['data_y'] = None
                eval_data['eval_time'] = None
                eval_data['Status'] = EvalFileStatus.Evaluating.value
                write_pickle(file_name=filename_evaluating, data=eval_data)
            return  # Since it is in completed status correctly, nothing more to do
    else:  # if there is no completed file then read from evaluating or waiting file and make evaluating file
        eval_data = read_pickle(file_name=(filename_waiting if not os.path.exists(filename_evaluating)
                                           else filename_evaluating))
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
    return


def run_dag_ga(run_type: str, dir_name: Optional[str] = None, file_name: Optional[str] = None):
    if run_type == 'suggest':
        ga_get_offsprings(dir_name)
    elif run_type == 'evaluate':
        run_ga_evaluator(filename=file_name)
    else:
        raise NotImplementedError
    return


def eval_init_individual_eval_files(dir_name_list: List[str], n_process: int, n_runs: Optional[int] = None):
    cmd_list = []
    for dir_name in dir_name_list:
        individual_ids = individual_ids_in_generation(dir_name=dir_name, generation=1)
        for indi_id in individual_ids:
            most_updated_filename = most_updated_eval_file_name(dir_name=dir_name, generation=1, individual_id=indi_id)
            if EvalFileStatus.Waiting.value in most_updated_filename:
                cmd_list.append(generate_bash_command(cmd_type='evaluate', file_name=most_updated_filename))
            elif EvalFileStatus.Evaluating.value in most_updated_filename:
                filename_waiting, _, _ = filenames_with_three_status(most_updated_filename)
                os.remove(most_updated_filename)
                cmd_list.append(generate_bash_command(cmd_type='evaluate', file_name=filename_waiting))
    if n_runs is not None:
        cmd_list = cmd_list[:n_process * n_runs]
    run_multiple_commands_in_parallel(cmd_list=cmd_list, n_process=n_process)
    return


def initialize_ga_from_init_data(dir_name: str, n_offsprings: int):
    individual_ids = individual_ids_in_generation(dir_name=dir_name, generation=1)
    data_x = None
    data_y = np.zeros((len(individual_ids),))
    eval_time = np.zeros((len(individual_ids),))
    evaluator = None
    for indi_id in individual_ids:
        file_name = most_updated_eval_file_name(dir_name=dir_name, generation=1, individual_id=indi_id)
        assert EvalFileStatus.Completed.value in file_name
        eval_data = read_pickle(file_name=file_name)
        eval_x = eval_data['data_x'].numpy()
        eval_y = eval_data['data_y']
        assert indi_id == eval_data['individual_id']
        if data_x is None:
            data_x = np.zeros((len(individual_ids), eval_x.size), dtype=np.int)
        data_x[indi_id] = eval_x
        data_y[indi_id] = eval_y
        eval_time[indi_id] = eval_data['eval_time']
        evaluator = eval_data['evaluator']
    pop = Population.new(X=data_x, F=data_y.reshape((-1, 1)))
    problem = DAGNMLPymooProblem(evaluator)
    generation = 1
    ga_data = \
        {'pop': pop, 'problem': problem, 'generation': generation, 'n_offsprings': n_offsprings, 'evaluator': evaluator}
    dir_name_time_tag = '_'.join([dir_name, generate_time_tag()])
    if not os.path.exists(os.path.join(DAG_EXP_DIR, dir_name_time_tag)):
        os.makedirs(os.path.join(DAG_EXP_DIR, dir_name_time_tag))
    save_ga(ga_data=ga_data, dir_name=dir_name_time_tag, exp_dir=DAG_EXP_DIR)
    return


def initialize_ga_bash_command(dir_name):
    cmd_parsed = []
    if HOSTNAME[:2] == 'fs':  # das5 server
        cmd_parsed += ['srun', '--time=00:15:00']
        if HOSTNAME in ['fs0', 'fs2', 'fs4']:
            cmd_parsed += ['-C', 'cpunode']
    elif HOSTNAME == 'DTA160000':
        pass
    else:
        raise NotImplementedError
    cmd_parsed += ['python', '-W', 'ignore', __file__, '--run_type', 'initga', '--dir_name', dir_name]
    return ' '.join(cmd_parsed)


def initialize_ga_in_parallel(dir_name_list: List[str], n_process: int):
    cmd_list = []
    for dir_name in dir_name_list:
        cmd_list.append(initialize_ga_bash_command(dir_name=dir_name))
    run_multiple_commands_in_parallel(cmd_list=cmd_list, n_process=n_process)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Structure Learning')
    parser.add_argument('--run_type', dest='run_type', type=str)
    parser.add_argument('--data', dest='data', type=str, default=None)
    parser.add_argument('--process', dest='process', type=int, default=None)
    parser.add_argument('--dir_file', dest='dir_file', type=str, default=None)
    parser.add_argument('--dir_name', dest='dir_name', type=str, default=None)
    parser.add_argument('--file_name', dest='file_name', type=str, default=None)
    parser.add_argument('--seed', dest='seed', type=int, default=None)

    GA_OPTIM_N_INITS = 100
    N_OFFSPRINGS = 20
    MAX_GEN = (1240 - GA_OPTIM_N_INITS) // N_OFFSPRINGS

    args = parser.parse_args()
    assert args.run_type in ['suggest', 'evaluate', 'jobmanager', 'initdata', 'initeval', 'initga']
    if args.run_type in ['suggest', 'evaluate']:
        run_dag_ga(run_type=args.run_type, dir_name=args.dir_name, file_name=args.file_name)
        sys.exit(0)
    elif args.run_type == 'jobmanager':
        assert args.process is not None
        with open(args.dir_file, 'rt') as f_:
            dir_name_list_ = f_.read().strip().split('\n')
        worker = JobManager(max_generation=MAX_GEN, n_parallel=args.process, dir_name_list=dir_name_list_)
        worker.run_jobs()
    elif args.run_type == 'initdata':
        assert args.data in ['sachs', 'child', 'insurance', 'alarm']
        evaluator_ = DAGNML(args.data)
        dir_name_ = generate_ga_exp_dirname(evaluator=evaluator_, batch_size=N_OFFSPRINGS, seed=args.seed,
                                            exp_dir=DAG_EXP_DIR, with_time_tag=False)
        set_init_data_x(evaluator=evaluator_, dir_name=dir_name_, seed=args.seed, n_init_data_x=GA_OPTIM_N_INITS)
    elif args.run_type == 'initeval':
        with open(args.dir_file, 'rt') as f_:
            dir_name_list_ = f_.read().strip().split('\n')
        eval_init_individual_eval_files(dir_name_list=dir_name_list_, n_process=args.process)
    elif args.run_type == 'initga':
        if args.dir_file is not None:
            assert args.dir_name is None
            with open(args.dir_file, 'rt') as f_:
                dir_name_list_ = f_.read().strip().split('\n')
            initialize_ga_in_parallel(dir_name_list=dir_name_list_, n_process=args.process)
        else:
            assert args.dir_name is not None
            initialize_ga_from_init_data(dir_name=args.dir_name, n_offsprings=N_OFFSPRINGS)

