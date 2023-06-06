import os, sys
import random
import itertools
import time
import copy
import functools
import pickle

import numpy as np
import tensorflow as tf
import networkx as nx
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab
import matplotlib.pyplot as plt
import netlsd
from sklearn import gaussian_process
from scipy import sparse
from bayes_opt import BayesianOptimization

import utils
sys.path.append('embedding_test/src/baseline/')
from GFPC import get_graph_feature

embedding_test_dir = 'embedding_test'
debug = True
cache = True

def split_graph(G, output_dir, radio=0.8):
    t_dir = output_dir
    Gs = G
    file_path = os.path.join(t_dir, 'graph.edgelist')
    file_test_path = os.path.join(t_dir, 'graph_test.edgelist')
    label_path = os.path.join(t_dir, 'label.txt')
    G_train = nx.Graph()
    G_test = nx.Graph()
    edges = np.random.permutation(list(Gs.edges()))
    nodes = set()
    for a, b in edges:
        if a not in nodes or b not in nodes:
            G_train.add_edge(a, b)
            nodes.add(a)
            nodes.add(b)
        else:
            G_test.add_edge(a, b)
    print(len(nodes), Gs.number_of_nodes())
    assert len(nodes) == Gs.number_of_nodes()
    assert len(nodes) == G_train.number_of_nodes()
    num_test_edges = int((1-radio)*Gs.number_of_edges())
    now_number = G_test.number_of_edges()
    if num_test_edges < now_number:
        test_edges = list(G_test.edges())
        G_train.add_edges_from(test_edges[:now_number-num_test_edges])
        G_test.remove_edges_from(test_edges[:now_number-num_test_edges])
    print("sample graph,origin: {} {}, train: {} {}, test: {} {}".format(Gs.number_of_nodes(), Gs.number_of_edges(), G_train.number_of_nodes(), G_train.number_of_edges(), G_test.number_of_nodes(), G_test.number_of_edges()))
    with utils.write_with_create(file_path) as f:
        for i, j in G_train.edges():
            print(i, j, file=f)
    with utils.write_with_create(file_test_path) as f:
        for i, j in G_test.edges():
            print(i, j, file=f)

def sample_graph(G, output_dir, s_n, times=10, with_test=False, radio=0.8, feature_path=None):
    if s_n is None:
        s_n = int(np.sqrt(G.number_of_nodes()))
    for t in range(times):
        t_dir = os.path.join(output_dir, 's{}'.format(t))
        n = random.randint(int(s_n/2), 2*s_n)
        Gs = utils.random_walk_induced_graph_sampling(G, n)
        mapping = dict(zip(Gs.nodes(), range(Gs.number_of_nodes())))
        if feature_path is not None:
            feats = sparse.load_npz(feature_path)
            row = []
            col = []
            data = []
            fr, fc = feats.nonzero()
            for i, j in zip(fr, fc):
                if i in mapping:
                    row.append(mapping[i])
                    col.append(j)
                    data.append(feats[i, j])
            feats = sparse.csr_matrix((data, (row, col)), shape=(len(mapping), feats.shape[1]))
        Gs = nx.relabel_nodes(Gs, mapping)
        file_path = os.path.join(t_dir, 'graph.edgelist')
        file_test_path = os.path.join(t_dir, 'graph_test.edgelist')
        label_path = os.path.join(t_dir, 'label.txt')
        feature_save_path = os.path.join(t_dir, 'features.npz')
        if feature_path is not None:
            utils.write_with_create(feature_save_path)
            sparse.save_npz(feature_save_path, feats)
        if not with_test:
            print("sample graph, nodes: {}, edges: {}, save into {}".format(Gs.number_of_nodes(), Gs.number_of_edges(), t_dir))
            with utils.write_with_create(file_path) as f:
                for i, j in Gs.edges():
                    print(i, j, file=f)
            with utils.write_with_create(label_path) as f:
                for i, data in Gs.nodes(data=True):
                    if 'label' in data:
                        for j in data['label']:
                            print(i, j, file=f)
        else:
            G_train = nx.Graph()
            G_test = nx.Graph()
            edges = np.random.permutation(list(Gs.edges()))
            nodes = set()
            for a, b in edges:
                if a not in nodes or b not in nodes:
                    G_train.add_edge(a, b)
                    nodes.add(a)
                    nodes.add(b)
                else:
                    G_test.add_edge(a, b)
            assert len(nodes) == Gs.number_of_nodes()
            assert len(nodes) == G_train.number_of_nodes()
            num_test_edges = int((1-radio)*Gs.number_of_edges())
            now_number = G_test.number_of_edges()
            if num_test_edges < now_number:
                test_edges = list(G_test.edges())
                G_train.add_edges_from(test_edges[:now_number-num_test_edges])
                G_test.remove_edges_from(test_edges[:now_number-num_test_edges])
            print("sample graph,origin: {} {}, train: {} {}, test: {} {}".format(Gs.number_of_nodes(), Gs.number_of_edges(), G_train.number_of_nodes(), G_train.number_of_edges(), G_test.number_of_nodes(), G_test.number_of_edges()))
            with utils.write_with_create(file_path) as f:
                for i, j in G_train.edges():
                    print(i, j, file=f)
            with utils.write_with_create(file_test_path) as f:
                for i, j in G_test.edges():
                    print(i, j, file=f)


def get_result(dataset_name, target_model, task, kargs, method='mle', sampled_dir='', debug=debug, cache=cache):
    rs = utils.RandomState()
    rs.save_state()
    rs.set_seed(0)
    embedding_filename = utils.get_names(target_model, **kargs)
    if task == 'classification':
        cf = os.path.abspath(os.path.join('result/{}'.format(dataset_name), sampled_dir, 'cf', embedding_filename))
    elif task == 'link_predict':
        cf = os.path.abspath(os.path.join('result/{}'.format(dataset_name), sampled_dir, 'lp', embedding_filename))
    embedding_filename = os.path.abspath(os.path.join('embeddings/{}'.format(dataset_name), sampled_dir, embedding_filename))
    dataset_filename = os.path.abspath(os.path.join('data/{}'.format(dataset_name), sampled_dir, 'graph.edgelist'))
    if target_model != 'gcn':
        if (not cache) or (not os.path.exists(embedding_filename)) or (os.path.getmtime(embedding_filename) < os.path.getmtime(dataset_filename)):
            utils.run_target_model(target_model, dataset_filename, os.path.dirname(embedding_filename), embedding_test_dir=embedding_test_dir, debug=debug, **kargs)
        if (not cache) or (not os.path.exists(cf)) or (os.path.getmtime(cf) < os.path.getmtime(embedding_filename)):
            if task == 'classification':
                labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename), 'label.txt'))
            elif task == 'link_predict':
                labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename)))
            utils.run_test(task, dataset_name, [embedding_filename], labels, cf, embedding_test_dir=embedding_test_dir)
    else:
        if (not cache) or (not os.path.exists(cf)):
            data_path = os.path.abspath(os.path.join('data/{}'.format(dataset_name)))
            with utils.cd(os.path.join(embedding_test_dir, 'src/baseline/gcn/gcn')):
                cmd = ('python3 main.py' +\
                        ' --epochs {} --hidden1 {} --learning_rate {}' +\
                        ' --output_filename {} --debug {} --dataset {} --input_dir {}').format(kargs['epochs'], kargs['hidden1'], kargs['learning_rate'], cf, debug, dataset_name, data_path)
                if debug:
                    print(cmd)
                else:
                    cmd += ' > /dev/null 2>&1'
                os.system(cmd)
    rs.load_state()
    res = np.loadtxt(cf, dtype=float)
    if len(res.shape) != 0:
        res = res[0]
    return res

def get_wne(dataset_name, sampled_dir='', cache=True, method='mle'):
    dataset_filename = os.path.abspath(os.path.join('data/{}'.format(dataset_name), sampled_dir, 'graph.edgelist'))
    labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename), 'label.txt'))
    save_path = os.path.abspath(os.path.join('embeddings/{}'.format(dataset_name), sampled_dir, 'wme_{}.embeddings'.format(method)))
    if (not cache) or (not os.path.exists(save_path)) or (os.path.getmtime(save_path) < os.path.getmtime(dataset_filename)):
        G = utils.load_graph(dataset_filename, label_name=None)
        do_full = (G.number_of_nodes()<10000)
        eigenvalues = 'full' if do_full else 'auto'
        if method == 'mle':
            wne = netlsd.heat(G, timescales=np.logspace(-2, 2, 10), eigenvalues=eigenvalues)
        else:
            wne = get_graph_feature.get_feature(G, method=method)
        with utils.write_with_create(save_path) as f:
            print(" ".join(map(str, wne)), file=f)
    return np.loadtxt(save_path)

def _get_mle_result(gp, dataset_name, target_model, task, without_wne, params, ps, s, X, y, method='mle'):
    wne = get_wne(dataset_name, '', cache=True, method=method) if not without_wne else None
    X_b_t, res_t = None, -1.0
    X_t = copy.deepcopy(X)
    y_t = copy.deepcopy(y)
    for i in range(s):
        X_b, y_b = gp.predict(ps, params.get_bound(ps), params.get_type(ps), wne)
        X_b = params.convert(X_b, ps)

        args = params.random_args(ps=ps, known_args=dict(zip(ps, X_b)))
        res = get_result(dataset_name, target_model, task, args, '')
        if res_t < res:
            res_t = res
            X_b_t = X_b
        if without_wne:
            X_b = [X_b]
        else:
            X_b = np.hstack((X_b, wne))
        X_t = np.vstack((X_t, X_b))
        y_t.append(res)
        gp.fit(X_t, y_t)
    X_b, y_b = gp.predict(ps, params.get_bound(ps), params.get_type(ps), wne)
    X_b = params.convert(X_b, ps)

    args = params.random_args(ps=ps, known_args=dict(zip(ps, X_b)))
    res = get_result(dataset_name, target_model, task, args, '')
    if res_t < res:
        res_t = res
        X_b_t = X_b
    return X_b_t, res_t

def dds_k(dataset_name, target_model, task, method='dds', sampled_number=5, without_wne=False, k=5, s=10, print_iter=10, debug=True):
    X = []
    NP = []
    y = []
    params = utils.Params(target_model)
    ps = params.arg_names
    info = []
    X_t, res_t = None, -1.0
    getting_sampled_result_time = 0.0

    sim = []
    if method == 'dds':
        o_wne = get_wne(dataset_name, '', method=method, cache=True)
        for t in range(sampled_number):
            wne = get_wne(dataset_name, 'sampled/s{}'.format(t), method=method, cache=True)
            sim.append(get_graph_feature.calc_similarity(o_wne, wne))
        total = sum(sim)
        sim = [x / total for x in sim]
        times = [int(sampled_number * k * x) for x in sim]
        rem = [sampled_number * k * x - int(sampled_number * k * x) for x in sim]
        rank = np.argsort(np.array(rem))
        for x in rank[-(sampled_number * k - sum(times)):]:
            times[x] += 1
        assert sum(times) == sampled_number * k
    
    max_sim = 0.0
    for t in range(sampled_number):
        b_t = time.time()
        wne = get_wne(dataset_name, 'sampled/s{}'.format(t), method=method, cache=True)
        tmp_performance = 0.0
        for v in range(times[t]):
            kargs = params.random_args(ps)
            res = get_result(dataset_name, target_model, task, kargs, 'sampled/s{}'.format(t))
            if without_wne:
                X.append([kargs[p] for p in ps])
            else:
                X.append([kargs[p] for p in ps])
                NP.append(wne)
            if debug:
                print('sample {}, {}/{}, kargs: {}, res: {}, time: {:.4f}s'.format(t, v, times[t], [kargs[p] for p in ps], res, time.time() - b_t))
            y.append(res)
            getting_sampled_result_time += time.time() - b_t

            if res > tmp_performance:
                tmp_performance = res
                tmp_kargs = kargs
        if sim[t] > max_sim:
            max_sim = sim[t]
            best_kargs = tmp_kargs

    if debug:
        print('total getting sampled result time: {:.4f}s'.format(getting_sampled_result_time))
    
    o_wne = get_wne(dataset_name, '', method=method, cache=True)
    dwr = utils.DWRRegressor(params.bound, o_wne)

    # with open('our_X.bin', 'rb') as fin:
    #     X = pickle.load(fin)

    # with open('our_y.bin', 'rb') as fin:
    #     y = pickle.load(fin)

    # with open('our_NP.bin', 'rb') as fin:
    #     NP = pickle.load(fin)
    tf.set_random_seed(0)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    res_best = 0.0
    total_time = getting_sampled_result_time

    # kargs = params.random_args(ps)
    base_params = [best_kargs[p] for p in ps]

    with tf.Session(config=config) as sess:    
        for t in range(s):
            start_time = time.time()

            dwr.build_graph(len(X), 0.005, 0.001)

            sess.run(tf.global_variables_initializer())

            dwr.fit_weight(sess, X, NP, y)
            dwr.fit_MLP(sess, X, NP, y)

            importance = np.sum(np.absolute(dwr.importance), axis=-1)
            rank = np.argsort(-importance)

            best_params = copy.deepcopy(base_params)

            for id in rank:
                if id < len(ps):
                    best_performance = 0.
                    curr_params = copy.deepcopy(best_params)

                    bound = params.get_bound([ps[id]])[0]
                    for param in np.linspace(bound[0], bound[1], 1000):
                        curr_params[id] = param
                        curr_performance = dwr.inference(sess, [curr_params], [o_wne])[0][0]

                        if curr_performance > best_performance:
                            best_performance = curr_performance
                            best_params = copy.deepcopy(curr_params)
                    
                    print(best_params)
            
            kargs = params.random_args(known_args=params.convert_dict(dict(zip(ps, best_params))))
            res_temp = get_result(dataset_name, target_model, task, kargs, '')
            X.append(best_params)
            NP.append(o_wne)
            y.append(res_temp)
            if res_temp > res_best:
                res_best = res_temp
                X_best = best_params
                base_params = best_params
                

            end_time = time.time()
            total_time += end_time - start_time
            info.append([res_temp, total_time])
            print('iters: {}/{}, params: {}, res: {}, time: {:.4f}s'.format(t, s, best_params, res_temp, total_time))

    np.set_printoptions(threshold=np.inf)
    ts = 'lp' if task == 'link_predict' else 'cf'
    with open('result/{}/our_X_{}_{}_{}.bin'.format(dataset_name, ts, method, target_model), 'wb') as fout:
        pickle.dump(X, fout)

    with open('result/{}/our_NP_{}_{}_{}.bin'.format(dataset_name, ts, method, target_model), 'wb') as fout:
        pickle.dump(NP, fout)

    with open('result/{}/our_y_{}_{}_{}.bin'.format(dataset_name, ts, method, target_model), 'wb') as fout:
        pickle.dump(y, fout)

    with open('result/{}/our_importance_{}_{}_{}.bin'.format(dataset_name, ts, method, target_model), 'wb') as fout:
        pickle.dump(importance, fout)

    if debug:
        print('final result: {}, time: {:.4f}s'.format(res_best, total_time))
        return X_best, res_best, info
    return X_best, res_best

def dds_test(dataset_name, target_model, task, method, color):
    print(dataset_name, target_model, task, method)
    params = utils.Params(target_model)
    ps = params.arg_names
    if target_model == 'AROPE':
        ps = ['1st order', '2nd order', '3rd order']
    elif target_model == 'deepwalk':
        ps = ['num walks', 'walk length', 'window size']
    elif target_model == 'gcn':
        ps = ['epochs', 'hidden1', 'learning rate', 'dropout', 'weight decay']
    
    o_wne = get_wne(dataset_name, '', method=method, cache=True)
    dwr = utils.DWRRegressor(params.bound, o_wne)

    ts = task

    if not os.path.exists('result/{}/our_X_{}_{}_{}.bin'.format(dataset_name, ts, method, target_model)):
        print('result/{}/our_X_{}_{}_{}.bin'.format(dataset_name, ts, method, target_model), 'NOT FOUND!!!')
        return

    with open('result/{}/our_X_{}_{}_{}.bin'.format(dataset_name, ts, method, target_model), 'rb') as fin:
        X = pickle.load(fin)

    with open('result/{}/our_y_{}_{}_{}.bin'.format(dataset_name, ts, method, target_model), 'rb') as fin:
        y = pickle.load(fin)

    with open('result/{}/our_NP_{}_{}_{}.bin'.format(dataset_name, ts, method, target_model), 'rb') as fin:
        NP = pickle.load(fin)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        dwr.build_graph(len(X), 0.005, 0.001)
        sess.run(tf.global_variables_initializer())
        dwr.fit_weight(sess, X, NP, y)
        dwr.fit_MLP(sess, X, NP, y)
        importance = np.sum(np.absolute(dwr.importance), axis=-1)[:len(ps)]
        importance /= np.sum(importance)
        
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.figure(figsize=(8, 6), dpi=80)
    ax = plt.subplot(1, 1, 1)
    ax.bar(ps, importance, color=color, alpha=0.8)
    ax.set_xlabel('parameter', fontsize=20)
    ax.set_ylabel('weight', fontsize=20)
    plt.grid()
    plt.xticks(fontsize=20, rotation=17)
    plt.yticks(fontsize=20)

    plt.savefig(os.path.join('weight', 'weight_{}_{}_{}.pdf'.format(target_model, ts, dataset_name)), bbox_inches='tight')

    print(dataset_name, target_model, ts, 'finished')


def mle_k(dataset_name, target_model, task='classification', method='mle', sampled_number=10, without_wne=False, k=16, s=0, print_iter=10, debug=False):
    X = []
    y = []
    params = utils.Params(target_model)
    ps = params.arg_names
    total_t = 0.0
    info = []
    X_t, res_t = None, -1.0
    if without_wne:
        gp = utils.GaussianProcessRegressor()
    else:
        K = utils.K(len(ps))
        gp = utils.GaussianProcessRegressor(K)
    getting_sampled_result_time = 0.0

    sim = []

    if method == 'mle':
        o_wne = get_wne(dataset_name, '', method=method, cache=True)
        for t in range(sampled_number):
            wne = get_wne(dataset_name, 'sampled/s{}'.format(t), method=method, cache=True)
            sim.append(get_graph_feature.calc_similarity(o_wne, wne))
        total = sum(sim)
        sim = [x / total for x in sim]
        times = [int(sampled_number * k * x) for x in sim]
        rem = [sampled_number * k * x - int(sampled_number * k * x) for x in sim]
        rank = np.argsort(np.array(rem))
        for x in rank[-(sampled_number * k - sum(times)):]:
            times[x] += 1
        assert sum(times) == sampled_number * k

    for t in range(sampled_number):
        b_t = time.time()
        i = t
        wne = get_wne(dataset_name, 'sampled/s{}'.format(i), method=method, cache=True)
        if method == 'mle_redispatch':
            loop = times[i]
        else:
            loop = k

        for v in range(loop):
            kargs = params.random_args(ps)
            res = get_result(dataset_name, target_model, task, kargs, 'sampled/s{}'.format(i))
            if without_wne:
                X.append([kargs[p] for p in ps])
            else:
                X.append(np.hstack(([kargs[p] for p in ps], wne)))
            if debug:
                print('sample {}, {}/{}, kargs: {}, res: {}, time: {:.4f}s'.format(t, v, k, [kargs[p] for p in ps], res, time.time()-b_t))
            y.append(res)
        getting_sampled_result_time += time.time() - b_t
    if debug:
        print('total getting sampled result time: {:.4f}s'.format(getting_sampled_result_time))

    total_t += getting_sampled_result_time

    for t in range(s):
        b_t = time.time()
        if t > 0 or not without_wne:
            gp.fit(np.vstack(X), y)
        X_temp, res_temp = _get_mle_result(gp, dataset_name, target_model, task, without_wne, params, ps, 0, X, y, method=method)
        if without_wne:
            X.append(X_temp)
        else:
            X.append(np.hstack((X_temp, wne)))
        y.append(res_temp)
        if res_t < res_temp:
            res_t = res_temp
            X_t = X_temp
        e_t = time.time()
        total_t += e_t-b_t
        info.append([res_temp, total_t])
        print('iters: {}/{}, params: {}, res: {}, time: {:.4f}s'.format(t, s, X_temp, res_temp, total_t))
    if debug:
        print('final result: {}, time: {:.4f}s'.format(res_t, total_t))
        return X_t, res_t, info
    return X_t, res_t

def random_search(dataset_name, target_model, task, k=16, debug=False, sampled_dir=''):
    X = []
    y = []
    params = utils.Params(target_model)
    ps = params.arg_names
    b_t = time.time()
    info = []
    for v in range(k):
        kargs = params.random_args(ps)
        #kargs = params.convert_dict(kargs, ps)
        if debug:
            print(kargs)
        res = get_result(dataset_name, target_model, task, kargs, sampled_dir)
        X.append([kargs[p] for p in ps])
        y.append(res)
        ind = np.argmax(y)
        total_t = time.time()-b_t
        if debug:
            info.append([y[ind], total_t])
        print('iters: {}/{}, params: {}, res: {}, time: {:.4f}s'.format(v, k, X[ind], y[ind], total_t))
    X = np.array(X)
    y = np.array(y)
    ind = np.argmax(y)
    if debug:
        return X[ind], y[ind], info
    return X[ind], y[ind]


def b_opt(dataset_name, target_model, task, k=16, debug=False, n_inits=0, inits=None, sampled_dir=''):
    params = utils.Params(target_model)
    ps = params.arg_names
    p_bound = dict(zip(ps, params.get_bound(ps)))
    def black_box_function(**kargs):
        b_t = time.time()
        x = [kargs[p] for p in ps]
        args = params.convert(x, ps)
        kargs = dict(zip(ps, args))
        kargs['emd_size'] = 64
        if target_model == 'AROPE':
            kargs['order'] = 3
        res = get_result(dataset_name, target_model, task, kargs, sampled_dir)
        e_t = time.time()
        print("############## params: {}, time: {}s".format(kargs, e_t-b_t))
        return res
    opt = BayesianOptimization(
            f=black_box_function,
            pbounds=p_bound,
            verbose=2)
    #opt.set_gp_params(normalize_y=False)
    if inits is not None:
        for d in inits:
            dd = dict(zip(ps, d))
            target = black_box_function(**dd)
            print(dd, target)
            opt.register(params=dd, target=target)
    opt.maximize(init_points=n_inits, n_iter=k)
    X = [opt.max['params'][p] for p in ps]
    y = opt.max['target']
    if debug:
        info = [res['target'] for res in opt.res]
        return X, y, info
    return X, y

def test_1(dataset_name, target_model, task):
    params = utils.Params(target_model)
    ps = params.arg_names
    b_t = time.time()
    info = []
    sampled_dir = 'sampled/s0'
    X = []
    y = []
    args = {'number-walks': 10, 'walk-length': 10, 'window-size': 3}
    temp_args = params.random_args(ps)
    res = get_result(dataset_name, target_model, task, temp_args, sampled_dir, cache=True)
    X.append([temp_args[p] for p in ps])
    y.append(res)
    print(i, j, [temp_args[p] for p in ps], res)
    return 0

def visualize_graph(output_dir, times=5):
    print('visualizing ...')
    for t in range(times):
        t_dir = os.path.join(output_dir, 's{}'.format(t))
        dataset_path = os.path.join(t_dir, 'graph.edgelist')
        save_path = os.path.join(output_dir, '{}.png'.format(str(t)))
        print('loading {} ...'.format(t_dir))
        G = utils.load_graph(dataset_path)
        print('{} loaded. {}'.format(t_dir, str(type(G))))

        nx.draw(G, node_size=5, width=0.5)
        plt.savefig(save_path)
        print('figure saved in {}'.format(save_path))


def main(args):
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    if len(args) == 0:
        dataset_name = 'pubmed'
        target_model = 'gcn'
        task = 'classification'
        ms = ['mle', 'random_search', 'b_opt']
    else:
        dataset_name = args[0]
        target_model = args[1]
        task = args[2]
        ms = args[3:]
    with_test = False
    dataset_path = 'data/{}/graph.edgelist'.format(dataset_name)
    label_path = 'data/{}/label.txt'.format(dataset_name)
    feature_path = None
    if task == 'link_predict':
        dataset_name = dataset_name+'_0.8'
        label_path = None
        with_test = True
    if target_model == 'gcn':
        feature_path = 'data/{}/features.npz'.format(dataset_name)
    if target_model == 'sample':
        G = utils.load_graph(dataset_path, label_path)
        b_t = time.time()
        split_graph(G, 'data/{}_0.8'.format(dataset_name), radio=0.8)
        sampled_number = 10#int(np.sqrt(G.number_of_nodes()))
        sample_graph(G, 'data/{}/sampled'.format(dataset_name), s_n=None, times=5, with_test=with_test, feature_path=feature_path)
        print('total sampling time: {:.4f}s'.format(time.time() - b_t))
        return 0
    ks = 5
    #test(dataset_name, target_model, task)
    sampled_dir = ''

#    visualize_graph('data/{}/sampled'.format(dataset_name), times=5)

    for m in ms:
        res = []
        for i in range(ks):
            info = []
            if m == 'mle':
                X, y, info = mle_k(dataset_name, target_model, task, method=m, sampled_number=5, without_wne=False, k=5, s=10, debug=True)
            elif m == 'mle_GFPC':
                X, y, info = mle_k(dataset_name, target_model, task, method=m, sampled_number=5, without_wne=False, k=5, s=10, debug=True)
            elif m == 'mle_GFPC54':
                X, y, info = mle_k(dataset_name, target_model, task, method=m, sampled_number=5, without_wne=False, k=5, s=10, debug=True)
            elif m == 'mle_w':
                X, y, info = mle_k(dataset_name, target_model, task, method=m, sampled_number=5, without_wne=True, k=5, s=10, debug=True)
            elif m == 'mle_redispatch':
                X, y, info = mle_k(dataset_name, target_model, task, method=m, sampled_number=5, without_wne=False, k=5, s=10, debug=True)
            elif m == 'dds':
                X, y, info = dds_k(dataset_name, target_model, task, method=m, sampled_number=5, k=5, debug=True)
            elif m == 'dds_test':
                methods = ['AROPE', 'AROPE', 'AROPE', 'AROPE', 'deepwalk', 'deepwalk', 'deepwalk', 'deepwalk', 'deepwalk', 'gcn']
                tasks = ['cf', 'cf', 'cf', 'lp', 'cf', 'cf', 'cf', 'lp', 'lp', 'cf']
                datasets = ['BlogCatalog', 'pubmed', 'Wikipedia', 'Wikipedia_0.8', 'BlogCatalog', 'pubmed', 'Wikipedia', 'BlogCatalog_0.8', 'Wikipedia_0.8', 'pubmed']
                colors = ['r', 'y', 'g', 'c', 'b', 'm', 'navy', 'lime', 'tan', 'gold']
                index = 9
                method = methods[index]
                task = tasks[index]
                dataset = datasets[index]
                ms = 'dds'
                color = colors[index]
                dds_test(dataset, method, task, ms, color)
            elif m == 'random_search':
                X, y, info = random_search(dataset_name, target_model, task, k=10, debug=True, sampled_dir=sampled_dir)
            elif m == 'random_search_l':
                X, y, info = random_search(dataset_name, target_model, task, k=5, debug=True, sampled_dir=sampled_dir)
            elif m == 'b_opt':
                b_t = time.time()
                X, y, info_t = b_opt(dataset_name, target_model, task, k=5, n_inits=5, debug=True, sampled_dir=sampled_dir)
                e_t = time.time()
                info = [[j, (e_t-b_t)/len(info_t)*(i+1)] for i, j in enumerate(info_t)]
            elif m == 'b_opt_l':
                b_t = time.time()
                X, y, info_t = b_opt(dataset_name, target_model, task, k=5, n_inits=1, debug=True, sampled_dir=sampled_dir)
                e_t = time.time()
                info = [[j, (e_t-b_t)/len(info_t)*(i+1)] for i, j in enumerate(info_t)]
            res.append(info)
            print(m, i, res)
            ts = 'lp' if task == 'link_predict' else 'cf'
            if sampled_dir == '':
                save_filename = 'result/{}/res_{}_{}_{}.npz'.format(dataset_name, ts, m, target_model)
            else:
                save_filename = 'result/{}/res_{}_{}_{}_{}.npz'.format(dataset_name, os.path.basename(sampled_dir), ts, m, target_model)
            np.savez(save_filename, res=res)
            print('Round {} of {} ended successfully.'.format(str(i), str(ks)))
            # with open('tmp.txt') as fout:
            #     print(res)

        print('final resul of', dataset_name, target_model, task, ms)
        # res_p = np.max(res[:,:,0], axis=-1)
        # res_t = np.sum(res[:,:,1], axis=-1)
        # idx = np.argmax(res_p)
        # print('final performance: {}, time: {} s'.format(str(res_p[idx]), str(res_t[idx])))


if __name__ == '__main__':
    main(sys.argv[1:])
