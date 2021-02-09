from collections import defaultdict
from functools import reduce
import numpy as np
import networkx as nx
import random
import os, sys
import tensorflow as tf
import six
import utils

def read_from_edgelist(filename, undirected=True, index_from_zero=True, with_head=False):
    line_num = 0
    graph = []
    with open(filename, 'r') as f:
        if with_head:
            l = f.readline().strip().split()
            total_n, total_e = int(l[0]), int(l[1])
            process_bar = utils.ShowProcess(total_e)
        for line in f:
            ls = line.strip().split()
            a, b = int(ls[0])+int(index_from_zero), int(ls[1])+int(index_from_zero)
            # print(a, b)
            while len(graph)-1 < max(a, b):
                graph.append([])
            if undirected:
                graph[a].append(b)
            graph[b].append(a)
            line_num += 1
            if with_head and line_num % 100000 == 0:
                process_bar.show_process(i=line_num)
    # print("nodes: {}, edges: {}".format(len(graph), line_num))
    return graph

def get_degree(graph):
    degree = [[] for _ in range(len(graph))]
    for i in range(len(graph)):
        degree[i] = len(graph[i])
    return degree

def sort_graph_by_degree(graph):
    degree = get_degree(graph)
    for i in range(len(graph)):
        graph[i] = sorted(graph[i], key=lambda x: degree[x])
    return graph

def get_max_degree(graph):
    return max([len(i) for i in graph])

def neighborhood_sum(*a):
    return list(map(lambda x: reduce(lambda y, z: y+z, x), zip(*a)))

def p_by_degree(x, graph):
    res = np.array([len(graph[i])+1e-2 for i in graph[x]], dtype=float)
    return res/np.sum(res)

def sampling_with_order(x, sampling=True, sampling_size=50, p=None, shuffle=False):
    # x: candidate neighbor ids
    if sampling and len(x) >= sampling_size:
        p = p / np.sum(p[np.array(x)-1])
        # print(np.sum(p), len(np.where(p>0)), len(x))
        if np.sum(p) == 0:
            p = np.ones(len(p))
            p = p / np.sum(p)
        # indices = np.random.choice(len(x), sampling_size, replace=False, p=p[np.array(x)-1])
        # indices = np.random.choice(len(x), sampling_size, replace=False, p=p)
        indices = np.random.choice(len(x), sampling_size, replace=False)
        indices = sorted(indices)
        if shuffle:
            random.shuffle(indices)
        return [x[i] for i in indices], sampling_size
    if len(x) < sampling_size:
        return x+[0]*(sampling_size-len(x)), len(x)
    # not sampling and len(x) > sampling size
    return x[:sampling_size], sampling_size


def get_neighborhood_list(node, graph, sim_mat, prob_mat=None, ns_size=10, sampling=True, sampling_size=100, shuffle=False):
    # p = p_by_degree(node, graph)
    if prob_mat is not None:
        p = prob_mat[node-1, :]
    else:
        p = None
    nodes, seqlen = sampling_with_order(graph[node], sampling, sampling_size, p=p, shuffle=shuffle)
    neg_nodes = np.random.choice(range(len(sim_mat)), ns_size, replace=False, p=sim_mat[node-1])
    neg_nodes += 1
    return node, nodes, seqlen, neg_nodes


def degree_to_class(degree, degree_max, nums_class):
    ind = int(np.log2(degree)/np.log2(degree_max+1)*nums_class)
    res = np.zeros(nums_class)
    res[ind] = 1
    return res

def next_batch(graph, sim_mat, prob_mat=None, degree_max=None, ns_size=10, sampling=False, sampling_size=100, args=None):
    if degree_max is None:
        degree_max = get_max_degree(graph)
    while True:
        l = np.random.permutation(range(1, len(graph)))
        for i in l:
            #yield (get_neighborhood(i, K, graph, padding=False, sampling=sampling, sampling_size=sampling_size), np.log(len(graph[i])))
            if len(graph[i]) == 0:
                continue
            yield (get_neighborhood_list(i, graph, sim_mat, prob_mat, ns_size=ns_size, sampling=sampling, sampling_size=sampling_size, shuffle=args.shuffle), np.log(len(graph[i])+1))

def read_network(filename, undirected=True):
    if not undirected:
        return nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
    return nx.read_edgelist(filename, nodetype=int)

def get_centrality(network, cent_type_='degree', undirected=True):
    if cent_type_ == 'degree':
        if undirected:
            return nx.degree_centrality(network)
        else:
            return nx.in_degree_centrality(network)
    elif cent_type_ == 'clustering':
        return nx.clustering(network)
    elif cent_type_ == 'closeness':
        return nx.closeness_centrality(network)
    elif cent_type_ == 'betweenness':
        return nx.betweenness_centrality(network)
    elif cent_type_ == 'eigenvector':
        return nx.eigenvector_centrality(network)
    elif cent_type_ == 'kcore':
        return nx.core_number(network)
    else:
        return None

'''
def save_centrality(data, file_path, type_):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    print("\n".join(map(str, data.values())), file=open(os.path.join(file_path, "{}.index".format(type_)), 'w'))
'''

def load_centrality(file_path, type_):
    filename = os.path.join(file_path, "{}.index".format(type_))
    data = np.loadtxt(filename)
    if type_ == 'spread_number':
        return data
    return np.vstack((np.arange(len(data)), data)).T

def save_to_tsv(labels, file_path, degree_max):
    res = []
    for type_ in labels:
        data = load_centrality(file_path, type_)
        if type_ == 'degree':
            data *= degree_max
        res.append(data)
    res = np.vstack(res).astype(int).T.tolist()
    with open(os.path.join(file_path, 'index.tsv'), 'w') as f:
        print("\t".join(labels), file=f)
        for data in res:
            print("\t".join(list(map(str, data))), file=f)
    return res

#  if __name__ == '__main__':
#     dataset_name = sys.argv[1]
#     undirected = sys.argv[2] == "True"
#     print("undirected", undirected)
#     G = read_network('dataset/{}.edgelist'.format(dataset_name), undirected)
#     G.remove_edges_from(G.selfloop_edges())
#     graph = read_from_edgelist('dataset/{}.edgelist'.format(dataset_name), index_from_zero=True, undirected=undirected)
#     graph = sort_graph_by_degree(graph)
#     data = get_neighborhood_list(2, graph)
#     save_path = 'result/{}/data'.format(dataset_name)
#     centrality = ['degree', 'closeness', 'betweenness', 'eigenvector', 'kcore']
#     centrality = ['degree', 'kcore']
#     for c in centrality:
#         save_centrality(get_centrality(G, c), save_path, c)
#     #save_to_tsv(['degree', 'kcore'], save_path, get_max_degree(graph))
#     print(load_centrality(save_path, 'degree'))

