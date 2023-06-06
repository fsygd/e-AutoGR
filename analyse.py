import argparse
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt


init_time = {
    'BlogCatalog_0.8_deepwalk_lp_dds': 44.47963500022888,
    
}

def get_data(dataset, task, ms, method):
    file_path = os.path.join(os.path.join('result', dataset), 'res_{}_{}_{}.npz'.format(task, ms, method))
    if not os.path.exists(file_path):
        return {
            'perf': [],
            'time': [],
            'trial': [],
            'std': 0,
        }
    npz = np.load(file_path)
    res = npz['res']
    print(dataset, task, ms, method, res.shape)
    if res.shape != (5, 10, 2):
        return {
            'perf': [],
            'time': [],
            'trial': [],
            'std': 0,
        }
    
    perf = res[:,:,0].tolist()
    for trial in range(5):
        for t in range(1, 10):
            perf[trial][t] = max(perf[trial][t - 1], perf[trial][t])
    perf = np.mean(np.asarray(perf), axis=0)

    time = res[:,:,1]
    time = np.mean(time, axis=0)
    time -= init_time.get("{}_{}_{}_{}".format(dataset, method, task, ms), 0)

    std = np.std(perf)

    return {
        'perf': perf,
        'time': time,
        'trial': np.arange(1, 11),
        'std': std,
    }


def make_figure(method='deepwalk', task='cf', dataset='BlogCatalog', fontsize=20, markersize=20, linewidth=4):
    if task in ['link_predict', 'lp']:
        task = 'lp'
        dataset += '_0.8'
    if task in ['classification', 'cf']:
        task = 'cf'
    
    data = {}

    for ms in ['dds', 'mle', 'random_search', 'b_opt']:
        data[ms] = get_data(dataset, task, ms, method)
    
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.set_xlabel('time(s)', fontsize=fontsize)
    ax1.set_ylabel('Micro-F1' if task == 'cf' else 'AUC', fontsize=fontsize)
    plt.xticks(rotation=17, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax1.plot(data['random_search']['time'], data['random_search']['perf'], 'b*-', label='Random', markersize=markersize, linewidth=linewidth)
    ax1.plot(data['b_opt']['time'], data['b_opt']['perf'], 'c*-', label='BayesOpt', markersize=markersize, linewidth=linewidth)
    ax1.plot(data['mle']['time'], data['mle']['perf'], 'r*-', label='AutoNE', markersize=markersize, linewidth=linewidth)
    ax1.plot(data['dds']['time'], data['dds']['perf'], 'y*-', label='e-AutoGR', markersize=markersize, linewidth=linewidth)
    ax1.legend()
    ax1.legend(loc=3, fontsize=fontsize)

    plt.savefig(os.path.join('figure', 'figure_{}_{}_{}_0.pdf'.format(method, task, dataset)), bbox_inches='tight')

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig = plt.figure(figsize=(8, 6), dpi=80)
    ax1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.set_xlabel('Performance', fontsize=fontsize)
    ax1.set_ylabel('# trials', fontsize=fontsize)
    ax1.yaxis.set_major_locator(plt.MultipleLocator(1))
    plt.xticks(rotation=17, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    ax1.plot(data['random_search']['perf'], data['random_search']['trial'], 'b*-', label='Random', markersize=markersize, linewidth=linewidth)
    ax1.plot(data['b_opt']['perf'], data['b_opt']['trial'], 'c*-', label='BayesOpt', markersize=markersize, linewidth=linewidth)
    ax1.plot(data['mle']['perf'], data['mle']['trial'], 'r*-', label='AutoNE', markersize=markersize, linewidth=linewidth)
    ax1.plot(data['dds']['perf'], data['dds']['trial'], 'y*-', label='e-AutoGR', markersize=markersize, linewidth=linewidth)
    ax1.legend()
    ax1.legend(loc=2, fontsize=fontsize)

    ax2 = fig.add_axes([0.6, 0.2, 0.25, 0.25])    
    plt.xticks(rotation=17)
    ax2.bar(['Random', 'BayesOpt', 'AutoNE', 'e-AutoGR'], np.hstack([data['random_search']['std'], data['b_opt']['std'], data['mle']['std'], data['dds']['std']]), color=['b', 'c', 'r', 'y'])
    ax2.set_title('std')
    plt.savefig(os.path.join('figure', 'figure_{}_{}_{}_1.pdf'.format(method, task, dataset)), bbox_inches='tight')

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--method', metavar='METHOD')
    # parser.add_argument('--task', metavar='TASK')
    # parser.add_argument('--dataset', metavar='DATASET')
    # args = parser.parse_args()
    methods = ['AROPE', 'AROPE', 'AROPE', 'AROPE', 'deepwalk', 'deepwalk', 'deepwalk', 'deepwalk', 'deepwalk', 'gcn']
    tasks = ['cf', 'cf', 'cf', 'lp', 'cf', 'cf', 'cf', 'lp', 'lp', 'cf']
    datasets = ['BlogCatalog', 'pubmed', 'Wikipedia', 'Wikipedia', 'BlogCatalog', 'pubmed', 'Wikipedia', 'BlogCatalog', 'Wikipedia', 'pubmed']

    for index in range(10):
        method = methods[index]
        task = tasks[index]
        dataset = datasets[index]
        make_figure(
            method=method,
            task=task,
            dataset=dataset,
        )
