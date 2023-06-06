import os
import utils
import networkx as nx
import numpy as np
from scipy import stats

def count_thv(graph):
    thv = []
    for i in range(len(graph)):
        out_degree = [graph.degree[j] for j in graph.neighbors(i)]
        thv.append((i, sum(out_degree) / len(out_degree)))
    return thv

def cal_lcs(graph):
    lcs = []
    for i in range(len(graph)):
        fai = 0
        nb1 = list(graph.neighbors(i))
        for j in nb1:
            nb2 = list(graph.neighbors(j))
            fai += len(set(nb1).union(set(nb2)))
        out_degree = graph.degree[i]
        if out_degree > 1:
            cv = 2 * fai / out_degree / (out_degree - 1)
        else:
            cv = 0
        lcs.append((i, cv))
    return lcs

def cal_acn(graph, lcs):
    acn = []
    for i in range(len(graph)):
        nbs = list(graph.neighbors(i))
        lcs_list = [lcs[j][1] for j in nbs]
        if len(lcs_list) > 0:
            acn.append((i, sum(lcs_list) / len(lcs_list)))
        else:
            acn.append((i, 0))
    return acn

def count_tri(graph):
    tri = nx.triangles(graph)
    return sum(tri.values())

def max_td(graph):
    total_degree = graph.degree
    return max(dict(total_degree).values())

def cal_gcc(graph, num_tri):
    beta = 0
    for i in range(len(graph)):
        num_nbs = len(list(graph.neighbors(i)))
        beta = beta + num_nbs*(num_nbs-1)/2
    gc = 3*num_tri/beta
    return gc

def get_feature(G, method='GFPC'):
    global_features = []

    graph_order = nx.Graph.order(G)
    global_features.append(graph_order)

    num_edges = nx.number_of_edges(G)
    global_features.append(num_edges)

    num_triangles = count_tri(G)
    global_features.append(num_triangles)

    global_clustering_coefficient = cal_gcc(G, num_triangles)
    global_features.append(global_clustering_coefficient)

    max_total_degree = max_td(G)
    global_features.append(max_total_degree)

    num_components = nx.number_connected_components(G)
    global_features.append(num_components)

    print('global feature dim: {}'.format(len(global_features)))
    if method in ['mle_GFPC', 'mle_redispatch', 'dds']:
        return global_features

    centrality = nx.eigenvector_centrality(G)
    pr = nx.pagerank(G, alpha=0.9)
    total_degree = G.degree
    two_hop = count_thv(G)
    local_clustering_score = cal_lcs(G)
    average_clustering_of_neighbors = cal_acn(G, local_clustering_score)

    centrality = np.asarray([centrality[i] for i in range(G.number_of_nodes())])
    pr = np.asarray([pr[i] for i in range(G.number_of_nodes())])
    total_degree = np.asarray([total_degree[i] for i in range(G.number_of_nodes())])
    two_hop = np.asarray([two_hop[i][1] for i in range(G.number_of_nodes())])
    local_clustering_score = np.asarray([local_clustering_score[i][1] for i in range(G.number_of_nodes())])
    average_clustering_of_neighbors = np.asarray([average_clustering_of_neighbors[i][1] for i in range(G.number_of_nodes())])

    node_features = np.stack((centrality, pr, total_degree, two_hop, local_clustering_score, average_clustering_of_neighbors))

    median = np.median(node_features, axis=-1)
    mean = np.mean(node_features, axis=-1)
    stdev = np.std(node_features, axis=-1)
    skewness = stats.skew(node_features, axis=-1)
    kurtosis = stats.kurtosis(node_features, axis=-1)
    variance = np.var(node_features, axis=-1)
    maxVal = stats.tmax(node_features, axis=-1)
    minVal = stats.tmin(node_features, axis=-1)

    node_features = np.concatenate((median, mean, stdev, skewness, kurtosis, variance, maxVal, minVal)).tolist()

    print('node feature dim: {}'.format(len(node_features)))
    
    return global_features + node_features

def calc_similarity(p, q, method='GFPC'):
	p = p.tolist()
	q = q.tolist()
	CD = 0
	for pi, qi in zip(p, q):
		CD += abs(pi - qi) / (abs(pi) + abs(qi))
	return 1 - CD / len(p)

def get_similarity(G1, G2, method='GFPC'):
	p = get_feature(G1, method=method)
	q = get_feature(G2, method=method)
	return cal_similarity(p, q)

if __name__ == '__main__':    

    dataset_name = 'BlogCatalog'
    sampled_dir=''
    cache=True
    dataset_filename = os.path.abspath(os.path.join('../../../../data/{}'.format(dataset_name), sampled_dir, 'graph.edgelist'))
    # labels = os.path.abspath(os.path.join(os.path.dirname(dataset_filename), 'label.txt'))
    save_path = os.path.abspath(os.path.join('../embeddings/{}'.format(dataset_name), sampled_dir, 'wme.embeddings'))
    if (not cache) or (not os.path.exists(save_path)) or (os.path.getmtime(save_path) < os.path.getmtime(dataset_filename)):
        G = utils.load_graph(dataset_filename, label_name=None)
        do_full = (G.number_of_nodes()<10000)
        eigenvalues = 'full' if do_full else 'auto'
        # wne = netlsd.heat(G, timescales=np.logspace(-2, 2, 10), eigenvalues=eigenvalues)
        
        centrality = nx.eigenvector_centrality(G)
        print('%s %0.2f'%(node,centrality[node]) for node in centrality)
    
        pr = nx.pagerank(G, alpha=0.9)
        print('%s %0.2f'%(node,pr[node]) for node in pr)
    
        print('\n======Degree======')
        total_degree = G.degree
        print(list(total_degree))
    
        # Two-Hop Away Neighbours
        print('\n======Two-Hop Away Neighbours======')
        two_hop = count_thv(G)
        print(two_hop)
    
        # Local Clustering Score
        print('\n======Local Clustering Score======')
        local_clustering_score = cal_lcs(G)
        print(local_clustering_score)
    
        # Average Clustering of Neighbourhood
        print('\n======Average Clustering of Neighbourhood======')
        average_clustering_of_neighbors = cal_acn(G, local_clustering_score)
        print(average_clustering_of_neighbors)
    
        graph_order = nx.Graph.order(G)
        print('graph order =', graph_order)
    
        num_edges = nx.number_of_edges(G)
        print('number of edges =', num_edges)
    
        num_triangles = count_tri(G)
        print('number of triangles =', num_triangles)
    
        global_clustering_coefficient = cal_gcc(G, num_triangles)
        print('global clustering coefficient =', global_clustering_coefficient)
    
        max_total_degree = max_td(G)
        print('max total degree =', max_total_degree)
    
        num_components = nx.number_connected_components(G)
        print('number of components =', num_components)
    
        # with utils.write_with_create(save_path) as f:
            # print(" ".join(map(str, wne)), file=f)
