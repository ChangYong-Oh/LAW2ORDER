from typing import List

import os
import dill
import pickle

import numpy as np

import matplotlib.pyplot as plt

import torch

from LAW2ORDER.experiments.config_file_path import EXP_DIR, REPO_DIR
from LAW2ORDER.experiments.tasks.ga_dag import DAGNMLPymooProblem


def get_last_filename(exp_dirname):
    last_file = max([elm for elm in os.listdir(exp_dirname) if elm[-4:] == '.pkl'])
    return last_file


def read_bo_exp_plot_data(exp_list, dirname):
    torch_opt = torch.min
    all_best_batch_eval_np = None
    n_init = None
    p_len = None
    for exp in exp_list:
        last_filename = get_last_filename(os.path.join(dirname, exp))
        with open(os.path.join(dirname, exp, last_filename), 'rb') as f:
            exp_data = pickle.load(f)
        batch_idx = exp_data['batch_idx']
        max_batch_idx = batch_idx.max().int()
        data_y = exp_data['data_y']
        best_batch_eval = data_y.new_zeros((max_batch_idx + 1, ))
        for i in range(max_batch_idx + 1):
            best_batch_eval[i] = torch_opt(data_y[batch_idx <= i])
        if all_best_batch_eval_np is None:
            all_best_batch_eval_np = best_batch_eval.reshape(-1, 1).numpy()
        else:
            new_max = min(all_best_batch_eval_np.shape[0], best_batch_eval.numel())
            all_best_batch_eval_np = np.hstack((all_best_batch_eval_np[:new_max],
                                                best_batch_eval.reshape(-1, 1).numpy()[:new_max]))
        if n_init is None:
            n_init = int(torch.sum(batch_idx == 0))
        else:
            assert n_init == int(torch.sum(batch_idx == 0))
        if p_len is None:
            p_len = exp_data['data_x'].size()[1]
        else:
            assert p_len == exp_data['data_x'].size()[1]
    return np.mean(all_best_batch_eval_np, axis=1), np.std(all_best_batch_eval_np, axis=1), \
           n_init, all_best_batch_eval_np.shape[1], p_len


def plot_exp(ax, plot_x, plot_y_mean, plot_y_std, n_runs, label, color, title):
    ax.plot(plot_x, plot_y_mean, color=color, label=label)
    stderr_scale = STD_SCALE / n_runs ** 0.5
    ax.fill_between(plot_x, plot_y_mean - stderr_scale * plot_y_std, plot_y_mean + stderr_scale * plot_y_std,
                    color=color, alpha=0.2)
    ax.set_title(title)


def group_benchmark_exp(dirname, problem_list, alg_list):
    exp_benchmark_group = dict()
    for exp in os.listdir(dirname):
        if not any([elm in exp for elm in problem_list]):
            continue
        benchmark, algorithm, batch_size = exp.split('_')[:3]
        batch_size = int(batch_size[1:])
        if benchmark in problem_list and algorithm in alg_list:
            try:
                exp_benchmark_group[benchmark][batch_size][algorithm].append(exp)
            except KeyError:
                try:
                    exp_benchmark_group[benchmark][batch_size][algorithm] = [exp]
                except KeyError:
                    try:
                        exp_benchmark_group[benchmark][batch_size] = {algorithm: [exp]}
                    except KeyError:
                        exp_benchmark_group[benchmark] = {batch_size: {algorithm: [exp]}}
    return exp_benchmark_group


def plot_benchmark_figure(benchmark_list: List[str], alg_list: List[str], dirname=EXP_DIR, legend=False):
    exp_benchmark_group = group_benchmark_exp(dirname=dirname, problem_list=benchmark_list, alg_list=alg_list)
    for benchmark in BENCHMARK_LIST:
        if benchmark not in exp_benchmark_group:
            continue
        for batch_size in BATCH_SIZE_LIST:
            if batch_size not in exp_benchmark_group[benchmark]:
                continue
            benchmark_group = exp_benchmark_group[benchmark][batch_size]
            fig, ax = plt.subplots()
            n_alg = 0
            for alg in alg_list:
                if alg not in benchmark_group:
                    continue
                exp_list = benchmark_group[alg]
                exp_mean, exp_std, n_init, n_runs, p_len = read_bo_exp_plot_data(exp_list, dirname)
                label, color = LABEL_COLOR_LIST[alg]
                plot_x = np.arange(int(n_init / batch_size), int(n_init / batch_size) + exp_mean.size)
                title = r'%s($S_{%d}$)' % (benchmark, p_len)
                plot_exp(ax, plot_x, exp_mean, exp_std, n_runs, label, color, title)
                n_alg += 1
                if legend and alg == 'BUCB':
                    ax.plot(np.mean(plot_x), np.mean(exp_mean), label=' ', color='w')
                    n_alg += 1
            ax.text(1.0, -0.02, r'$(\times %d)$' % batch_size,
                    horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
            fig.tight_layout()
            if legend:
                ax.legend(loc='upper center', ncol=n_alg // 2, fontsize=6)
                plt.savefig(os.path.join(FIG_SAVE_DIR, '%s-supp.png' % benchmark))
            else:
                plt.savefig(os.path.join(FIG_SAVE_DIR, '%s-main.png' % benchmark))
            plt.close(fig)


def plot_benchmark_legend(alg_list):
    fig, ax = plt.subplots(figsize=(8.5,5))
    for alg in alg_list:
        if alg is None:
            label, color = ' ', 'w'
        else:
            label, color = LABEL_COLOR_LIST[alg]
        plt.plot(0, 0, label=label, color=color)
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.legend(ncol=len(alg_list) // 2, loc='center')
    plt.savefig(os.path.join(FIG_SAVE_DIR, '%s.png' % 'benchmark_legend'))


def read_ga_exp_plot_data(exp_list, dirname):
    numpy_opt = np.min
    all_best_batch_eval_np = None
    n_init = None
    p_len = None
    for exp in exp_list:
        batch_size = int(exp.split('_')[2][1:])
        last_gen = int(os.path.splitext(get_last_filename(os.path.join(dirname, exp)))[0].split('_')[-1])
        best_batch_eval = []
        with open(os.path.join(dirname, exp, '%s_001.pkl' % exp), 'rb') as f:
            pop_001 = dill.load(f)['pop']
            p_len = pop_001[0].get('X').size
        for i in range(batch_size, len(pop_001) + 1, batch_size):
            best_batch_eval.append(numpy_opt([pop_001[elm].get('F') for elm in range(i)]))

        for g in range(2, last_gen + 1):
            with open(os.path.join(dirname, exp, '%s_%03d.pkl' % (exp, g)), 'rb') as f:
                pop_g = dill.load(f)['pop']
            best_batch_eval.append(
                min(best_batch_eval[-1], numpy_opt([pop_g[elm].get('F') for elm in range(len(pop_g))])))
        best_batch_eval = np.array(best_batch_eval)

        if all_best_batch_eval_np is None:
            all_best_batch_eval_np = best_batch_eval.reshape(-1, 1)
        else:
            new_max = min(all_best_batch_eval_np.shape[0], best_batch_eval.size)
            all_best_batch_eval_np = np.hstack((all_best_batch_eval_np[:new_max],
                                                best_batch_eval.reshape(-1, 1)[:new_max]))
        if n_init is None:
            n_init = batch_size
        else:
            assert n_init == batch_size
    return np.mean(all_best_batch_eval_np, axis=1), np.std(all_best_batch_eval_np, axis=1), \
           n_init, all_best_batch_eval_np.shape[1], p_len


def group_structure_exp(dirname, problem_list, alg_list):
    exp_structure_group = dict()
    for exp in os.listdir(dirname):
        if not any([elm in exp for elm in problem_list]):
            continue
        if len(exp.split('_')) < 6:
            continue
        benchmark, algorithm, batch_size = exp.split('_')[:3]
        batch_size = int(batch_size[1:])
        if benchmark in problem_list and algorithm in alg_list:
            try:
                exp_structure_group[benchmark][batch_size][algorithm].append(exp)
            except KeyError:
                try:
                    exp_structure_group[benchmark][batch_size][algorithm] = [exp]
                except KeyError:
                    try:
                        exp_structure_group[benchmark][batch_size] = {algorithm: [exp]}
                    except KeyError:
                        exp_structure_group[benchmark] = {batch_size: {algorithm: [exp]}}
    return exp_structure_group


def plot_structure(structure_list: List[str], alg_list: List[str],
                   dirname=os.path.join(os.path.split(EXP_DIR)[0], 'LAW2ORDERDAG'),
                   which='main'):
    assert which in ['main', 'supp']
    exp_structure_group = group_structure_exp(dirname=dirname, problem_list=structure_list, alg_list=alg_list)
    batch_size = 20
    bo_rounds = 31
    for structure in STRUCTURE_LIST:
        if structure not in exp_structure_group:
            continue
        structure_group = exp_structure_group[structure][batch_size]
        fig, ax = plt.subplots()
        for alg in alg_list:
            if alg not in structure_group:
                continue
            exp_list = structure_group[alg]
            if alg == 'GA':
                exp_mean, exp_std, n_init, n_runs, p_len = read_ga_exp_plot_data(exp_list, dirname)
            else:
                exp_mean, exp_std, n_init, n_runs, p_len = read_bo_exp_plot_data(exp_list, dirname)
            label, color = LABEL_COLOR_LIST[alg]
            plot_x = np.arange(int(n_init / batch_size), int(n_init / batch_size) + exp_mean.size)
            if which == 'main':
                plot_x = plot_x[:bo_rounds]
                exp_mean = exp_mean[:bo_rounds]
                exp_std = exp_std[:bo_rounds]
            title = r'%s($S_{%d}$)' % (structure.split('-')[1].upper(), p_len)
            plot_exp(ax, plot_x, exp_mean, exp_std, n_runs, label, color, title)
        ax.text(1.0, -0.02, r'$(\times %d)$' % batch_size,
                horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
        fig.tight_layout()
        if which == 'supp':
            ax.legend(loc='upper right', fontsize=10)
            plt.savefig(os.path.join(FIG_SAVE_DIR, '%s-supp.png' % structure.replace('NML', '')))
        else:
            plt.savefig(os.path.join(FIG_SAVE_DIR, '%s-main.png' % structure.replace('NML', '')))
        plt.close(fig)


def plot_structure_legend(alg_list):
    fig, ax = plt.subplots(figsize=(8.5,5))
    for alg in alg_list:
        label, color = LABEL_COLOR_LIST[alg]
        plt.plot(0, 0, label=label, color=color)
    fig.tight_layout()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.legend(ncol=len(alg_list), loc='center')
    plt.savefig(os.path.join(FIG_SAVE_DIR, '%s.png' % 'DAG_legend'))


BENCHMARK_LIST = ['QAP-chr12a', 'QAP-nug22', 'QAP-esc32a',
                  'FSP-car5', 'FSP-hel2', 'FSP-reC19',
                  'TSP-burma14', 'TSP-bayg29', 'TSP-att48',]
ALG_BENCHMARK_MAIN = ['BUCB', None, 'DPP-posterior-MAX-est', 'DPP-posterior-SAMPLE-est',
                       'MACE-ucb', 'MACE-est', 'qPOINT-ei', 'qPOINT-est',
                       'WDPP-posterior-MAX-ei', 'WDPP-posterior-MAX-est']
ALG_BENCHMARK_SUPP = ['BUCB', None, 'DPP-posterior-MAX-est', 'DPP-posterior-SAMPLE-est',
                       'MACE-ucb', 'MACE-est', 'qPOINT-ei', 'qPOINT-est',
                       'WDPP-prior-MAX-ei', 'WDPP-prior-MAX-est', 'WDPP-posterior-MAX-ei', 'WDPP-posterior-MAX-est']
STRUCTURE_LIST = ['DAGNML-sachs', 'DAGNML-child', 'DAGNML-insurance', 'DAGNML-alarm']
ALG_STRUCTURE = ['GA', 'qPOINT-ei', 'qPOINT-est', 'WDPP-posterior-MAX-est']
LABEL_COLOR_LIST = {'BUCB': ('BUCB', 'tab:gray'),
                    'DPP-posterior-MAX-est': ('DPP-MAX-EST', 'tab:cyan'),
                    'DPP-posterior-SAMPLE-est': ('DPP-MAX-SAMPLE', 'tab:olive'),
                    'MACE-ucb': ('MACE-UCB', 'tab:brown'),
                    'MACE-est': ('MACE-EST', 'tab:purple'),
                    'qPOINT-ei': ('q-EI', 'k'),
                    'qPOINT-est': ('q-EST', 'b'),
                    'WDPP-prior-MAX-ei': ('LAW-prior-EI', 'r'),
                    'WDPP-prior-MAX-est': ('LAW-prior-EST', 'tab:orange'),
                    'WDPP-posterior-MAX-ei': ('LAW-EI', 'g'),
                    'WDPP-posterior-MAX-est': ('LAW-EST', 'm'),
                    'GA': ('GA', 'teal')}
BATCH_SIZE_LIST = [5, 10, 20]
STD_SCALE = 1
FIG_SAVE_DIR = os.path.join(REPO_DIR, 'LAW2ORDER', 'experiments', 'figures')

if __name__ == '__main__':
    plot_benchmark_legend(alg_list=ALG_BENCHMARK_MAIN)
    plot_benchmark_figure(benchmark_list=BENCHMARK_LIST, alg_list=ALG_BENCHMARK_MAIN, legend=False)
    plot_benchmark_figure(benchmark_list=BENCHMARK_LIST, alg_list=ALG_BENCHMARK_SUPP, legend=True)
    plot_structure_legend(alg_list=ALG_STRUCTURE)
    plot_structure(structure_list=STRUCTURE_LIST, alg_list=ALG_STRUCTURE, which='main')
    plot_structure(structure_list=STRUCTURE_LIST, alg_list=ALG_STRUCTURE, which='supp')
