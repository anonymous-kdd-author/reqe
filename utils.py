import numpy as np
import operator
import tensorflow as tf
import scipy
import networkx as nx
import sys, time, os
import pandas as pd
import graph
import random

def load_from_wv_format(filename):
    with open(filename) as f:
        l = f.readline().split()
        total_num, embedding_size = int(l[0]), int(l[1])
        ls = list(map(lambda x: x.strip().split(), f.readlines()))
        total_num = max([int(line[0]) for line in ls])+1
        res = np.zeros((total_num, embedding_size), dtype=float)
        for line in ls:
            res[int(line[0])] = list(map(float, line[1:]))
    return res

def save_as_wv_format(filename, data):
    with open(filename, 'w') as f:
        nums, embedding_size = data.shape
        print(nums, embedding_size, file=f)
        for j in range(nums):
            print(j, *data[j], file=f)

def load_embeddings(filename, file_type=None):
    #print("load embeddings ", filename)
    if file_type is None:
        file_type = filename.strip().split('.')[-1]
    if file_type == 'embeddings':
        return load_from_wv_format(filename)
    elif file_type == 'npy':
        return np.load(filename)
    else:
        print('unsupported file type!')
        return None

def MSE(x, y):
    return np.mean((x-y)**2)

def print_array(X):
    a, b = X.shape
    print("\n".join(["\t".join(["{:.6e}"]*b)]*a).format(*X.flatten()))

def regularize_dataset(filename, output_filename, uniq=False, sort=True, label_filename=None, label_output_filename=None):
    data = np.loadtxt(filename).astype(int)
    if uniq:
        data = np.array(list(filter(lambda x: x[0] < x[1], data)))
    data_ids = list(set(data.flatten()))
    if sort:
        data_ids.sort()
    mapping = dict(zip(data_ids, range(len(data_ids))))
    res = np.vectorize(mapping.get)(data)
    np.savetxt(output_filename, res, fmt='%d')
    if label_filename is not None:
        labels = np.loadtxt(label_filename).astype(int)
        res_label_id = np.vectorize(mapping.get)(labels[:, 0])
        res_label = np.vstack((res_label_id, labels[:, 1])).T
        np.savetxt(label_output_filename, res_label, fmt='%d')

def init_embedding(degrees, degree_max, emb_size_all, cent_feat=[]):
    # print(len(cent_feat))
    emb_size = int(emb_size_all/(1+len(cent_feat)))
    degree_emb = np.vstack([np.random.normal(i*1.0/degree_max, 0.0001, (emb_size_all - emb_size * len(cent_feat))) for i in degrees])
    print(degree_emb.shape)
    if len(cent_feat) > 0:
        cent_embs = []
        for cent in cent_feat:
            # print(cent)
            cent = np.concatenate([[0], cent])
            cent_emb = np.vstack([np.random.normal(i*1.0/np.max(cent), 0.0001, emb_size) for i in cent])
            cent_emb *= np.max(degree_emb) / np.max(cent_emb)
            cent_embs.append(cent_emb)
        cent_embs = np.concatenate(cent_embs, axis=1)
        print(cent_embs.shape)
        degree_emb = np.concatenate([degree_emb, cent_embs], axis=1)
    return degree_emb

def selu(x):
    """Scaled Exponential Linear Unit. (Klambauer et al., 2017)
    # Arguments
        x: A tensor or variable to compute the activation function for.
    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.contrib.keras.activations.elu(x, alpha)

def gather_col(X, batch_size, l):
    if type(l) == list or type(l) == tuple:
        return tf.gather_nd(X, [[[i, j] for j in l] for i in range(batch_size)])
    else:
        return tf.gather_nd(X, [[i, l] for i in range(batch_size)])

def VS(A, alpha, iter_num=100):
    assert 0 < alpha < 1
    assert type(A) is scipy.sparse.csr.csr_matrix
    lambda_1 = scipy.sparse.linalg.eigsh(A, k=1, which='LM', return_eigenvectors=False)[0]
    n = A.shape[0]
    d = np.array(A.sum(1)).flatten()
    d_inv = np.diag(1./d)
    dsd = np.random.normal(0, 1/np.sqrt(n), (n, n))
    #dsd = np.zeros((n, n))
    I = np.eye(n)
    for i in range(iter_num):
        dsd = alpha/lambda_1*A.dot(dsd)+I
        if i % 10 == 0:
            print('VS', i, '/', iter_num)
    return d_inv.dot(dsd).dot(d_inv)

def generate_graph(N, d, p):
    return nx.random_graphs.watts_strogatz_graph(N, d, p)




def load_graphsage_embedding(data_dir, result_dir):
    X = np.load(os.path.join(data_dir, 'val.npy'))
    ind = np.loadtxt(os.path.join(data_dir, 'val.txt'), dtype=int)
    emd_size = X.shape[1]
    X = X[ind]
    np.save(os.path.join(result_dir, 'baseline_{}/graphsage.npy'.format(emd_size)), X)


def load_node_labels(data_dir):
    data = pd.read_csv(data_dir, header=None, sep=' ')
    labels = np.array(data[1])
    return labels


def construct_rpr_matrix(G, k_RPR=20):
    '''
    Construct Rooted PageRank matrix
    '''
    # print("Number of nodes: {}".format(len(G.nodes())))
    num_walks = len(G.nodes()) * 100
    num_nodes = len(G.nodes())
    max_node = np.max(list(G.nodes()))+1

    # print("Number of walks: {}".format(num_walks))
    # print("Walking...")
    walk_times = 10
    walk_length = 50
    alpha = 0.5
    walks = graph.build_deepwalk_corpus(G, num_paths=walk_times, path_length=walk_length,
                                        alpha=alpha, rand=random.Random(32))
    all_counts = np.zeros((max_node, max_node))
    for node in walks.keys():
        walks_n = walks[node]
        for walk in walks_n:
            all_counts[node][walk] += 1
    all_counts = np.array([i/np.sum(i) * 10 for i in all_counts])
    where_are_NaNs = np.isnan(all_counts)
    all_counts[where_are_NaNs] = 0
    rpr_matrix = np.asarray(all_counts, dtype='double')
    rpr_matrix = rpr_matrix.transpose()
    # print("after transpose:", rpr_matrix)
    rpr_matrix = np.array([sorted(i)[::-1][:k_RPR] for i in rpr_matrix])
    return rpr_matrix



