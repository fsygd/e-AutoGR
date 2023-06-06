import os
import argparse

import numpy as np


def parse(dataset, method, task, ms):
    if task == 'link_predict':
        task = 'lp'
        dataset += '_0.8'
    else:
        task = 'cf'

    npz = np.load(os.path.join(os.path.join('result', dataset), 'res_{}_{}_{}.npz'.format(task, ms, method)))
    res = npz['res']
    assert res.shape == (5, 10, 2)
    p = res[:,:,0]
    p = np.max(p, axis=-1)

    t = res[:,:,1]
    t = np.sum(t, axis=-1)

    idx = np.argmax(p)

    print(dataset, method, task, ms)

    print('best performance: {},{}s'.format(format((p[idx]), '.4f'), format(t[idx], '.2f')))
    print('average performance: {},{}s'.format(format(np.mean(p), '.4f'), format(np.mean(t), '.2f')))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', metavar='DATASET')
    parser.add_argument('--method', metavar='METHOD')
    parser.add_argument('--task', metavar='TASK')
    parser.add_argument('--ms', metavar='ms')
    args = parser.parse_args()

    parse(
        dataset=args.dataset,
        method=args.method,
        task=args.task,
        ms=args.ms,
    )
