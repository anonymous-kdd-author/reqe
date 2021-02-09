from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.multiclass import OneVsRestClassifier
import argparse
import numpy as np
import time
import torch
import network, utils
import networkx as nx
from pyREQE import pyREQE
import time
import collections
import six
import os
import tqdm
import pandas as pd
import graph as gr
import scipy.io as sio
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances, silhouette_score




os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)


def parse_args():
    parser = argparse.ArgumentParser("Reqe: Representation-based regular equivalence")
    parser.add_argument('--data_path', type=str, help='Directory to load data.')
    parser.add_argument('--prob_path', type=str, help='Directory to probability matrix.')
    parser.add_argument('--label_path', type=str, help='Directory to load node labels.')
    parser.add_argument('--save_path', type=str, help='Directory to save data.')
    parser.add_argument('--save_suffix', type=str, default='eni', help='Directory to save data.')
    parser.add_argument('-s', '--embedding_size', type=int, default=16, help='the embedding dimension size')
    parser.add_argument('-e', '--epochs_to_train', type=int, default=40, help='Number of epoch to train. Each epoch processes the training data once completely')
    parser.add_argument('-b', '--batch_size', type=int, default=512, help='Number of training examples processed per step')
    parser.add_argument('-k', '--krpr', type=int, default=20,
                        help='Number of training examples processed per step')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--undirected', type=bool, default=True, help='whether it is an undirected graph')
    parser.add_argument('-a', '--alpha', type=float, default=0.3, help='the rate of structure loss and negative sampling loss')
    parser.add_argument('-l', '--lamb', type=float, default=0.3, help='the rate of structure loss and regularization loss')
    parser.add_argument('-g', '--grad_clip', type=float, default=1.0, help='clip gradients')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('-K', type=int, default=1, help='K-neighborhood')
    parser.add_argument('--sampling_size', type=int, default=10, help='sample number')
    parser.add_argument('--negative_samples', type=int, default=10, help='sample number')
    parser.add_argument('--seed', type=int, default=-1, help='random seed')
    parser.add_argument('--graphs', type=int, default=4, help='number of graphs')
    parser.add_argument('--index_from_0', type=bool, default=True, help='whether the node index is from zero')
    parser.add_argument('--sampling', action='store_true', default=False, help='sample neighbors')
    parser.add_argument('--shuffle', action='store_true', default=False, help='sample neighbors')
    parser.add_argument('--agg', action='store_true', default=False, help='use aggregated graph')
    parser.add_argument('--avg', action='store_true', default=False, help='use aggregated graph')
    parser.add_argument('--pretrained', action='store_true', default=False, help='whether the node index is from zero')
    return parser.parse_args()



def evaluation(node_emb_all):
    # print(node_emb.shape, labels.shape)
    # acc_val_all, acc_test_all = [], []
    # names = ['GPA', 'Major']
    # for i, label_path in enumerate(['label_gpa.txt', 'label_majors_top.txt']):
    Y = np.loadtxt(args.label_path).astype(int)
    # print("evaluating on {} nodes".format(len(Y)))
    labels = Y[:, 1]
    nodes_id = Y[:, 0]
    node_emb = node_emb_all[nodes_id]
    acc_test, acc_val = [], []
    ratio = 0.6
    th_val = int(ratio * len(labels))
    th_test = int(0.8 * len(labels))
    for _ in range(50):
        index = np.random.permutation(range(len(labels)))
        clf = LogisticRegression(random_state=0, multi_class='ovr', solver='lbfgs', max_iter=500).fit(
            node_emb[index[:th_val]], labels[index[:th_val]])
        acc_test.append(clf.score(node_emb[index[th_test:]], labels[index[th_test:]]))
        acc_val.append(clf.score(node_emb[index[th_val:th_test]], labels[index[th_val:th_test]]))
    print("val acc: {:.4f} ({:.4f}), test acc: {:.4f} ({:.4f})".format(np.mean(acc_val), np.std(acc_val),
                                                                           np.mean(acc_test), np.std(acc_test)),
          flush=True)
    acc_val_all = np.mean(acc_val)
    acc_test_all = np.mean(acc_test)
    return acc_val_all, acc_test_all


def read_embedding_file(filename):
    with open(filename) as f:
        num_node, size = f.readline().split(' ')
        emb = np.zeros((6306, int(size)))
        for i in range(int(num_node)):
            l = np.array(f.readline().split(' ')).astype(float)
            k = int(l[0])
            emb[k, :] = l[1:]
    return emb



def train_model(args, graph_view, init_emb):
    model = pyREQE(num_nodes=len(graph_view), args=args, hidden_dim=args.embedding_size, init_emb=init_emb)
    if args.cuda:
        model = model.cuda()
    node_emb = model.get_embedding()
    if args.label_path is not None:
        acc_val, acc_test = evaluation(node_emb)
        best_acc = [acc_test, 0]
        best_val = [acc_val, 0]
        best_emb = node_emb
        best_epoch = -1
    else:
        pass
    model.train()
    total_num = int((len(graph) - 1) / args.batch_size)
    if total_num < 16:
        args.batch_size = 2 ** int(np.log(len(graph) - 1) / np.log(2) - 2)
    total_num = int((len(graph) - 1) / args.batch_size)
    for e in tqdm.tqdm(range(args.epochs_to_train)):
        for i in range(total_num):
            inputs = [[] for _ in range(5)]
            data_batch = network.next_batch(graph, sim_mat, prob_mat, ns_size=args.negative_samples, sampling=args.sampling, sampling_size=args.sampling_size, args=args)
            for _ in range(args.batch_size):
                data, label = six.next(data_batch)
                data += (label,)
                for input_, d in zip(inputs, data):
                    input_.append(d)
            struc_loss, guided_loss, ns_loss = model.train_step(inputs)
        if args.label_path is not None:
            node_emb = model.get_embedding()
            acc_val, acc_test = evaluation(node_emb)
            if acc_val > best_val[0]:
                best_acc = [acc_test, 0]
                best_val = [acc_val, 0]
                best_emb = node_emb
                best_epoch = e
        else:
            print("epoch {}, structure loss: {:.6f}, guided loss: {:.6f}, ns loss: {:.6f}".format(
                e + 1, struc_loss, guided_loss, ns_loss))
    print("Best acc = {:.4f} at epoch {}".format(best_acc[0], best_acc[1], best_epoch), flush=True)
    return best_emb


def get_prob_from_edgelist(edgelist, num_nodes):
    res = np.zeros((num_nodes, num_nodes))
    for l in edgelist:
        [u,v] = [int(i) for i in l.split(' ')]
        res[u,v] += 1
        res[v,u] += 1
    res[np.isnan(res)] = 0
    return res


if __name__ == '__main__':
    args = parse_args()
    args.cuda = torch.cuda.is_available()
    np.random.seed(int(time.time()) if args.seed == -1 else args.seed)
    start_time = time.time()

    if args.data_path is not None:
        G = gr.load_edgelist(args.data_path)
        if 'dblp' in args.data_path:
            num_nodes_all = 17191
        else:
            prob_mat = np.loadtxt(args.prob_path)
            num_nodes_all = len(prob_mat)
        print("number of nodes:", num_nodes_all)
        graph = network.read_from_edgelist(args.data_path, index_from_zero=args.index_from_0)
        graph = [np.unique(l) for l in graph]
        network.sort_graph_by_degree(graph)
        degrees = [len(j) for j in graph]
        degree_max = network.get_max_degree(graph)
        rpr_mat = utils.construct_rpr_matrix(G, args.krpr)
        sim_mat = cosine_similarity(rpr_mat)
        sim_mat = 1 - sim_mat
        sim_mat = normalize(sim_mat, axis=1)
        row_sums = sim_mat.sum(axis=1)
        sim_mat = sim_mat / row_sums[:, np.newaxis]
        for i in range(len(sim_mat)):
            for j in range(len(sim_mat)):
                if sim_mat[i, j] <= 0:
                    sim_mat[i, j] = 0
        for cents in [['closeness', 'betweenness']]:
            print("centrality: ", cents, flush=True)
            edgelist = []
            degrees_by_graph = []
            graph_id = list(range(args.graphs))
            try:
                cent_feats = np.load(args.prob_path[:-4] + "_cent_feats_" + ','.join(cents) + ".npy")
                cent_feats_list = np.mean(cent_feats, axis=0).transpose()
                print("loaded centrality!", flush=True)
                for gi in tqdm.tqdm(graph_id[:args.graphs]):
                    G_curr = nx.read_edgelist(args.data_path + '.' + str(gi))
                    degrees_by_graph.append(
                        [G_curr.degree(str(v)) if str(v) in G_curr.nodes() else 0 for v in range(num_nodes_all)])
                    edgelist.extend(nx.generate_edgelist(G_curr, data=False))
            except:
                cent_feats = []
                print("cannot find cent file", flush=True)
                for gi in tqdm.tqdm(graph_id[:args.graphs]):
                    G_curr = nx.read_edgelist(args.data_path + '.' + str(gi))
                    degrees_by_graph.append(
                        [G_curr.degree(str(v)) if str(v) in G_curr.nodes() else 0 for v in range(num_nodes_all)])
                    edgelist.extend(nx.generate_edgelist(G_curr, data=False))
                    graph_cent = np.zeros((num_nodes_all, len(cents)))
                    for i, cent in enumerate(cents):
                        kvs = np.array(list(network.get_centrality(G_curr, cent).items()))
                        graph_cent[kvs[:, 0].astype(int), i] = kvs[:, 1]
                    cent_feats.append(graph_cent)
                if len(cents) > 0:
                    cent_feats = np.array(cent_feats)
                    # cent_feats = np.load(args.prob_path[:-4] + "_cent_feats_" + str(args.graphs) + ".npy")
                    cent_feats = cent_feats[graph_id[:args.graphs]]
                    f = open(args.prob_path[:-4] + "_cent_feats_" + ','.join(cents) + ".npy", 'wb')
                    np.save(f, cent_feats)
                    print("Computed struc. properties in {:.4f} minutes".format((time.time() - start_time) / 60))
                    cent_feats_list = np.mean(cent_feats, axis=0).transpose()
                else:
                    cent_feats_list = []
            degrees_by_graph = np.sum(np.array(degrees_by_graph), axis=0)
            degrees_by_graph = [0] + list(degrees_by_graph)
            degrees, degree_max = degrees_by_graph, np.max(degree_max)
            prob_mat = get_prob_from_edgelist(edgelist, num_nodes_all)

        init_emb = utils.init_embedding(degrees, degree_max, args.embedding_size, cent_feats_list)
        print("Initialized embedding in {:.4f} minutes".format((time.time() - start_time) / 60))
        start_time = time.time()

        emb = train_model(args, graph_view=graph, init_emb=init_emb)
        print("emb size: ", emb.shape, flush=True)
        print("Learned embedding in {:.4f} minutes".format((time.time() - start_time) / 60), flush=True)
        filepath = args.data_path[6:-4]
        np.savetxt(filepath + "emb_" + ",".join(cents) + '.txt', emb)
