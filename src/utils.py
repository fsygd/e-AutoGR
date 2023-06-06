import random
import os, sys
import collections
import itertools
import pickle as pkl
import time
import scipy.sparse as sp

import networkx as nx
import tensorflow as tf
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn import gaussian_process
from scipy.optimize import minimize
from bayes_opt import BayesianOptimization

def rand(size, a, b, decimals=4):
    res = np.random.random_sample(size)*(b-a)+a
    if decimals is not None:
        return np.around(res, decimals=decimals)
    return res

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def split_network(G, N):
    sc = SpectralClustering(N, affinity='precomputed')
    return sc.fit_predict(nx.adjacency_matrix(G))

def random_walk_induced_graph_sampling(G, N, T=100, growth_size=2, n_starts=5):
    # Refer to https://github.com/Ashish7129/Graph-Sampling
    G = nx.convert_node_labels_to_integers(G, 0, 'default', True)
    for n, data in G.nodes(data=True):
        G.node[n]['id'] = n
    n_node = G.number_of_nodes()
    labels = nx.get_node_attributes(G, 'label')
    candidate_set = set()
    if len(labels) > 0:
        candidate_set_value = set()
        l = np.random.permutation(list(labels.keys()))
        for i in l:
            for j in labels[i]:
                if j not in candidate_set_value:
                    candidate_set.add(i)
                    candidate_set_value.add(j)
    while len(candidate_set) < n_starts:
        candidate_set.add(np.random.random_integers(n_node))
    candidate_set = list(candidate_set)[:n_starts]
    sampled_nodes = set()
    for i, temp_node in enumerate(candidate_set):
        sampled_nodes.add(G.node[temp_node]['id'])
        iter_ = 1
        nodes_before_t_iter = 0
        curr_node = temp_node
        N_t = int(N/n_starts)*(i+1)
        while len(sampled_nodes) < N_t:
            edges = [n for n in G.neighbors(curr_node)]
            index_of_edge = random.randint(0, len(edges)-1)
            chosen_node = edges[index_of_edge]
            sampled_nodes.add(G.node[chosen_node]['id'])
            curr_node = chosen_node
            iter_ += 1
            if iter_ % T == 0:
                if (len(sampled_nodes)-nodes_before_t_iter < growth_size):
                    curr_node = random.randint(0, n_node-1)
                nodes_before_t_iter = len(sampled_nodes)
    sampled_graph = G.subgraph(sampled_nodes)
    return sampled_graph

def generate_mask(dataset_path, radio=0.8):
    G = load_graph(os.path.join(dataset_path, 'graph.edgelist'), os.path.join(dataset_path, 'label.txt'))
    labels = nx.get_node_attributes(G, 'label')
    l = np.random.permutation(list(labels.keys()))
    n = int(radio*len(l))
    with open(os.path.join(dataset_path, 'label_mask_train'), 'w') as f:
        for i in l[:n]:
            print(i, file=f)
    with open(os.path.join(dataset_path, 'label_mask_test'), 'w') as f:
        for i in l[n:]:
            print(i, file=f)


def write_with_create(path):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return open(path, 'w')

def load_graph(edgelist_filename, label_name=None):
    G = nx.read_edgelist(edgelist_filename, nodetype=int)
    if label_name is not None:
        labels = np.loadtxt(label_name, dtype=int)
        ### multi-label
        l = collections.defaultdict(list)
        for i, j in labels:
            l[i].append(j)
        ### Warning:: The call order of arguments `values` and `name` switched between v1.x & v2.x.
        nx.set_node_attributes(G, l, 'label')
    print("load graph", G.number_of_nodes(), G.number_of_edges())
    return G

def run_target_model(method, input_filename, output_dir, embedding_test_dir, debug=True, **kargs):
    sys.path.append(embedding_test_dir)
    from src.baseline import baseline
    with cd(embedding_test_dir):
        baseline(method, None, kargs['emd_size'], input_filename, output_dir, debug=debug, **kargs)

def run_test(task, dataset_name, models, labels, save_filename, embedding_test_dir):
    sys.path.append(embedding_test_dir)
    from src.test import test
    args = {}
    if task == 'classification':
        args['radio'] = [0.8]
        args['label_name'] = labels
        evalution = None
    elif task == 'link_predict':
        evalution = 'AUC'
        args['data_dir'] = labels
    args['sampling_mapping'] = {'Flickr': 100000, 'wiki': 1000000}

    with cd(embedding_test_dir):
        test(task, evalution, dataset_name, models, save_filename=save_filename, **args)

def get_names(method, **args):
    kargs = args
    if method == 'node2vec':
        embedding_filename = os.path.join("{}_{:d}_{:d}_{:d}_{:d}_{:.4f}_{:.4f}".format(method, kargs['emd_size'], kargs['num-walks'], kargs['walk-length'], kargs['window-size'], kargs['p'], kargs['q']))
    elif method == 'deepwalk':
        embedding_filename = os.path.join("{}_{:d}_{:d}_{:d}_{:d}".format(method, kargs['emd_size'], kargs['number-walks'], kargs['walk-length'], kargs['window-size']))
    elif method == 'gcn':
        #embedding_filename = os.path.join("{}_{:d}_{:d}_{:.4f}".format(method, kargs['epochs'], kargs['hidden1'], kargs['learning_rate']))
        embedding_filename = os.path.join("{}_{:d}_{:d}_{:.4f}_{:.4f}_{:.4f}".format(method, kargs['epochs'], kargs['hidden1'], kargs['learning_rate'], kargs['dropout'], kargs['weight_decay']))
    elif method == 'AROPE':
        embedding_filename = os.path.join("{}_{}_".format(method, kargs['emd_size'])+'_'.join(['{:.4f}'.format(kargs['w{}'.format(i+1)]) for i in range(kargs['order'])]))
    return embedding_filename

def random_with_bound_type(bound, type_):
    res = []
    for b, t in zip(bound, type_):
        if t == int:
            res.append(random.randint(*b))
        elif t == float:
            res.append(rand(1, *b)[0])
        else:
            res.append(None)
    return res


def find_b_opt_max(gp, ps, p_bound, p_type, w=None, n_warmup=100000, n_iter=100):
    """
    refer to acq_max https://github.com/fmfn/BayesianOptimization/blob/master/bayes_opt/util.py
    """
    X = []
    for k in range(n_warmup):
        X.append(random_with_bound_type(p_bound, p_type))
    if w is not None:
        X = np.hstack((X, np.tile(w, (len(X), 1))))
    y = gp.predict(X)
    ind = np.argmax(y)
    x_max, y_max = X[ind][:len(ps)], y[ind]
    temp_w = [] if w is None else w
    def utility(x, kappa=2.576):
        mean, std = gp.predict([list(x)+list(temp_w)], return_std=True)
        #print("######### mean, std ", x, mean, std)
        return (mean + kappa*std)[0]
    for i in range(n_iter):
        x_try = random_with_bound_type(p_bound, p_type)
        res = minimize(lambda x: -utility(x),
                        x_try,
                        bounds=p_bound,
                        method='L-BFGS-B')
        if not res.success:
            continue
        if -res.fun >= y_max:
            x_max = res.x
            y_max = -res.fun

    return x_max, y_max

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

class K(gaussian_process.kernels.Kernel):
    def __init__(self, n=3):
        self.n = n
        self.kernels = [gaussian_process.kernels.Matern(nu=2.5), gaussian_process.kernels.Matern(nu=2.5)]

    def __call__(self, X, Y=None):
        n = self.n
        if Y is None:
            Y = X
        return self.kernels[0](X[:, :n], Y[:, :n])*self.kernels[1](X[:, n:], Y[:, n:])

    def diag(self, X):
        n = self.n
        return self.kernels[0].diag(X[:, :n])*self.kernels[1].diag(X[:, n:])

    def is_stationary(self):
        return np.all([kernel.is_stationary() for kernel in self.kernels])

class GaussianProcessRegressor(object):
    def __init__(self, kernel=None):
        if kernel is None:
            kernel = gaussian_process.kernels.Matern(nu=2.5)
        self.gp = gaussian_process.GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=10)

    def fit(self, X, y):
        self.gp.fit(X, y)

    def predict(self, ps, p_bound, type_, w=None):
        return find_b_opt_max(self.gp, ps, p_bound, type_, w)

class DWRRegressor(object):
    def __init__(self, X_bound, NP_bound):
        self.X_lowerbound = [x[0] for x in X_bound]
        self.X_gap = [x[1] - x[0] for x in X_bound]
        self.NP_lowerbound = [0] * len(NP_bound)
        self.NP_gap = NP_bound
        self.NP_gap[3] = 1.
        self.p = len(X_bound)
        self.p_np = len(NP_bound)

    def build_graph(self, n, lr1, lr2):
        p = self.p
        p_np = self.p_np

        # for global balancing

        self.balance_X = tf.placeholder(tf.float32, [None, p], 'balance_X')
        self.balance_NP = tf.placeholder(tf.float32, [None, p_np], 'balance_NP')
        self.balance_G = tf.Variable(tf.ones([n, 1]))
        
        with tf.variable_scope("global_balancing", reuse=tf.AUTO_REUSE):
            self.loss_balancing = tf.constant(0, tf.float32)
            for j in range(1, p + 1):
                X_j_and_NP = tf.concat([tf.slice(self.balance_X, [j * n, 0], [n, p]), self.balance_NP], 1)
                T = tf.slice(self.balance_X, [0, j - 1], [n, 1])
                balancing_j = tf.divide(tf.matmul(tf.transpose(self.balance_G * self.balance_G),tf.matmul(T, tf.cast(np.ones((1, p + p_np)), tf.float32)) * X_j_and_NP), tf.constant(n, tf.float32)) - tf.divide(tf.matmul(tf.transpose(self.balance_G * self.balance_G), T), tf.reduce_sum(self.balance_G * self.balance_G)) * tf.divide(tf.matmul(tf.transpose(self.balance_G * self.balance_G), X_j_and_NP), tf.constant(n, tf.float32))
                self.loss_balancing += tf.norm(balancing_j, ord=2)
            
            self.loss_weight_sum = (tf.reduce_sum(self.balance_G * self.balance_G) - n) ** 2
            self.loss_weight_l2 = tf.reduce_sum((self.balance_G * self.balance_G) ** 2)
            
            self.balance_loss = 2000.0 / p * self.loss_balancing + 0.0005 * self.loss_weight_sum + 0.00005 * self.loss_weight_l2
            self.balance_optimizer = tf.train.RMSPropOptimizer(lr1).minimize(self.balance_loss)

        # for weighted regression

        self.reg_X = tf.placeholder(tf.float32, [None, p + p_np], 'reg_X')
        self.reg_Y = tf.placeholder(tf.float32, [None], 'reg_Y')
        self.reg_W = tf.placeholder(tf.float32, [None, 1], 'reg_W')

        with tf.variable_scope('weighted_regression', reuse=tf.AUTO_REUSE):
            self.MLP = tf.layers.dense(self.reg_X, 5, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='mlp')
            with tf.variable_scope('mlp', reuse=True):
                self.reg_importance = tf.get_variable('kernel')

            self.relu = tf.nn.relu(self.MLP)
            self.hypothesis = tf.layers.dense(self.relu, 1, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='fc')

            self.reg_loss = tf.divide(tf.reduce_sum(self.reg_W * (self.reg_Y - self.hypothesis) ** 2), tf.reduce_sum(self.reg_W))
            self.reg_optimizer = tf.train.RMSPropOptimizer(lr2).minimize(self.reg_loss)
        
        self.saver = tf.train.Saver()
        print('build finished')
    
    def _global_balancing(self, sess, X_in, NP_in, num_steps, tol):
        display_step = 1000
        p = self.p
        
        X_feed = X_in
        for j in range(p):
            X_j = np.copy(X_in)
            X_j[:,j] = 0
            X_feed = np.vstack((X_feed,X_j))
        
        l_pre = 0
        for i in range(1, num_steps+1):
            _, l, l_balancing, l_weight_sum, l_weight_l2 = sess.run(
                [
                    self.balance_optimizer,
                    self.balance_loss,
                    self.loss_balancing,
                    self.loss_weight_sum,
                    self.loss_weight_l2,
                ],
                feed_dict={
                    self.balance_X: X_feed,
                    self.balance_NP: NP_in
                }
            )

            if abs(l-l_pre) <= tol:
                print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f ... %f' % (i, l, l_balancing, l_weight_sum, l_weight_l2))
                break
            l_pre = l
            if i % display_step == 0 or i == 1:
                print('Converge ... Step %i: Minibatch Loss: %f ... %f ... %f ... %f' % (i, l, l_balancing, l_weight_sum, l_weight_l2))
                
        Weight = sess.run([self.balance_G * self.balance_G])
        
        return  Weight[0]
    
    def fit_weight(self, sess, X_in, NP_in, y_in):
        start_time = time.time()
        if not isinstance(X_in, np.ndarray):
            X_in = np.asarray(X_in)
        if not isinstance(NP_in, np.ndarray):
            NP_in = np.asarray(NP_in)
        if not isinstance(y_in, np.ndarray):
            y_in = np.asarray(y_in)

        X_in = (X_in - self.X_lowerbound) / self.X_gap
        NP_in = (NP_in - self.NP_lowerbound) / self.NP_gap
        X_in = X_in * 4 - 2
        NP_in = NP_in * 4 - 2

        num_steps = 20000; tol = 1e-8
        self.weight = self._global_balancing(sess, X_in, NP_in, num_steps, tol)
    
        print('fit weight finished, time: {}s'.format(time.time() - start_time))

    def _weighted_regression(self, sess, is_train, X_in, y_in=None, num_steps=None, tol=None):
        display_step = 1000

        if is_train:
            l_pre = 0
        
            for i in range(1, num_steps + 1):
                _, l = sess.run(
                    [
                        self.reg_optimizer,
                        self.reg_loss,
                    ],
                    feed_dict={
                        self.reg_X: X_in,
                        self.reg_W: self.weight,
                        self.reg_Y: y_in
                    }
                )

                if abs(l - l_pre) <= tol:
                    print('Converge ... Step %i: Minibatch Loss: %f' % (i, l))
                    break
                l_pre = l
                if i % display_step == 0 or i == 1:
                    print('Converge ... Step %i: Minibatch Loss: %f' % (i, l))

            if not os.path.exists('./models'):
                os.makedirs('./models')
            self.saver.save(sess, 'models/weighted_regression.ckpt')
            print('weighted regression finished')
        else:
            # saver.restore(sess, 'models/weighted_regression.ckpt')
            pass

        return sess.run(
            [
                self.reg_importance,
                self.hypothesis,
            ],
            feed_dict={
                self.reg_X: X_in,
                self.reg_W: self.weight
            }
        )

    def fit_MLP(self, sess, X_in, NP_in, y_in):
        start_time = time.time()
        if not isinstance(X_in, np.ndarray):
            X_in = np.asarray(X_in)
        if not isinstance(NP_in, np.ndarray):
            NP_in = np.asarray(NP_in)
        if not isinstance(y_in, np.ndarray):
            y_in = np.asarray(y_in)

        X_in = (X_in - self.X_lowerbound) / self.X_gap
        NP_in = (NP_in - self.NP_lowerbound) / self.NP_gap
        X_in = X_in * 4 - 2
        NP_in = NP_in * 4 - 2

        X_in = np.hstack((X_in, NP_in))
        
        num_steps = 3000; tol = 1e-8
        self.importance, _ = self._weighted_regression(sess, True, X_in, y_in, num_steps, tol)
        print('fit MLP finished, time: {}s'.format(time.time() - start_time))

    def inference(self, sess, X_in, NP_in):
        start_time = time.time()
        if not isinstance(X_in, np.ndarray):
            X_in = np.asarray(X_in)
        if not isinstance(NP_in, np.ndarray):
            NP_in = np.asarray(NP_in)

        X_in = (X_in - self.X_lowerbound) / self.X_gap
        NP_in = (NP_in - self.NP_lowerbound) / self.NP_gap
        X_in = X_in * 4 - 2
        NP_in = NP_in * 4 - 2

        X_in = np.hstack((X_in, NP_in))

        _, hypothesis = self._weighted_regression(sess, False, X_in)

        return hypothesis

class meta_learner(BayesianOptimization):
    def set_kernel(kernel):
        self._gp.kernel = kernel

class RandomState(object):
    def __init__(self):
        self.state = None

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def save_state(self):
        self.state = (random.getstate(), np.random.get_state())

    def load_state(self):
        random.setstate(self.state[0])
        np.random.set_state(self.state[1])

class Params(object):
    def __init__(self, method):
        eps = 1e-6
        self.method = method
        if method == 'node2vec':
            self.arg_names = ['num-walks', 'walk-length', 'window-size', 'p', 'q']
            self.type_ = [int, int, int, float, float]
            self.bound = [(2, 20), (2, 80), (2, 10), (0.0001, 2), (0.0001, 2)]
        elif method == 'deepwalk':
            self.arg_names = ['number-walks', 'walk-length', 'window-size']
            self.type_ = [int, int, int]
            self.bound = [(2, 20), (2, 80), (2, 20)]
        elif method == 'gcn':
            self.arg_names = ['epochs', 'hidden1', 'learning_rate', 'dropout', 'weight_decay']
            self.type_ = [int, int, float, float, float]
            self.bound = [(10, 300), (2, 64), (0.0001, 0.1), (0.1, 0.9), (1e-4, 100e-4)]
        elif method == 'AROPE':
            n = 3
            self.arg_names = ['w{}'.format(i+1) for i in range(n)]
            self.type_ = [float for _ in range(n)]
            self.bound = [(0, 3) for _ in range(n)]
        self.ind = dict(zip(self.arg_names, range(len(self.arg_names))))

    def get_type(self, ps=None):
        if ps is None:
            return self.type_
        return [self.type_[self.ind[p]] for p in ps]

    def get_bound(self, ps=None):
        if ps is None:
            return self.bound
        return [self.bound[self.ind[p]] for p in ps]

    def convert(self, X, ps=None):
        type_ = self.get_type(ps)
        bound = np.array(self.get_bound(ps))
        X = np.clip(X, bound[:, 0], bound[:, 1])
        res = []
        for x, t in zip(X, type_):
            if t == int:
                res.append(int(round(x, 0)))
            elif t == float:
                res.append(round(x, 4))
        return res

    def convert_dict(self, d, ps=None):
        if ps is None:
            ps = self.arg_names
        for p in ps:
            x = np.clip(d[p], self.bound[self.ind[p]][0], self.bound[self.ind[p]][1])
            t = self.type_[self.ind[p]]
            if t == int:
                x = int(round(x, 0))
            elif t == float:
                x = round(x, 4)
            d[p] = x
        return d

    def random_args(self, ps=None, emd_size=64, known_args={}):
        if ps is None:
            ps = self.arg_names
        type_ = self.get_type(ps)
        bound = self.get_bound(ps)
        res = random_with_bound_type(bound, type_)
        d = dict(zip(ps, res))
        for arg in known_args:
            d[arg] = known_args[arg]
        if self.method != 'gcn':
            d['emd_size'] = emd_size
        if self.method == 'AROPE':
            d['order'] = 3
        return d

def analysis_result(data_dir):
    fs = os.listdir(data_dir)
    fs = np.array([np.loadtxt(os.path.join(data_dir, f)) for f in fs if not f.endswith('names')])
    print(fs.shape)
    scale = 100
    d = (fs[:, 0]*scale).astype(int)
    for k, v in collections.Counter(d).most_common():
        print(k*1.0/scale, v, "{:.2f}".format(v*1.0/fs.shape[0]))

def check_label(data_dir):
    fn = os.path.join(data_dir, 'label.txt')
    d = np.loadtxt(fn, dtype=int)
    c = collections.Counter(d[:, 1])
    for k, v in c.most_common():
        print(k, v)

def convert_gcn_data(dataset_name, input_dir, output_dir):
    def parse_index_file(filename):
        """Parse index file."""
        index = []
        for line in open(filename):
            index.append(int(line.strip()))
        return index
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(os.path.join(input_dir, "ind.{}.{}".format(dataset_name, names[i])), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(os.path.join(input_dir, "ind.{}.test.index".format(dataset_name)))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_name == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    print(G.number_of_nodes(), G.number_of_edges())
    with open('data/{}/label.txt'.format(dataset_name), 'w') as f:
        for i in range(len(labels)):
            assert sum(labels[i])>0
            for j, k in enumerate(labels[i]):
                if k == 1:
                    print(i, j, file=f)

    with open('data/{}/graph.edgelist'.format(dataset_name), 'w') as f:
        for i, j in G.edges():
            print(i, j, file=f)

    sp.save_npz('data/{}/features.npz'.format(dataset_name), features.tocsr())

if __name__ == '__main__':
    #analysis_result('result/BlogCatalog/cf/')
    #check_label('data/citeseer/sampled/s1/')
    for i in range(5):
        generate_mask('data/pubmed/sampled/s{}'.format(i))
    #convert_gcn_data('pubmed', 'embedding_test/src/baseline/gcn/gcn/data/', 'data/')
