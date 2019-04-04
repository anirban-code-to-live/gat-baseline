from __future__ import print_function

import os

import numpy as np
import networkx as nx

from utils import common_utils

def read_graph(dataset, is_weighted, is_directed, num_classes):
    '''
    Reads the graph structure using networkx.
    # Arguments
        dataset: Name of the dataset
        is_weighted: set to true if the graph is weighted
        is_directed: set to true if the graph is directed
        num_classes: integer denoting number of unique labels for the dataset
    # Returns
        A networkx graph object
    '''
    print("Loading the data...")
    FILE_PATH = os.path.abspath(__file__)
    MODULE_PATH = os.path.dirname(FILE_PATH)
    SRC_PATH = os.path.dirname(MODULE_PATH)
    PROJ_PATH = os.path.dirname(SRC_PATH)
    DATA_PATH = os.path.join(PROJ_PATH, 'data/{}/'.format(dataset))

    # Read edgelist file
    with open("{}{}.edgelist".format(DATA_PATH, dataset), 'rb') as edgelist_file:
        if is_weighted:
            graph = nx.read_edgelist(edgelist_file, nodetype = int, data = (('weight',float),), create_using = nx.DiGraph())
        else:
            graph = nx.read_edgelist(edgelist_file, nodetype = int, create_using = nx.DiGraph())
            # set all weights as default weight = 1
            for edge in graph.edges():
                graph[edge[0]][edge[1]]['weight'] = 1

    edgelist_file.close()

    # set if graph should be directed or undirected
    if not is_directed:
        graph = graph.to_undirected()

    # Read labels file
    labels = []
    with open("{}{}_label.csv".format(DATA_PATH, dataset), 'r') as f:
        for line in f:
            labels.append(common_utils.sample_mask(int(line), num_classes, np.int))
    labels = np.asarray(labels)

    return graph, labels


def print_graph(graph):
    '''
    Prints the graph using networkx module
    # Arguments
        graph: A networkx graph object
    '''
    matplotlib.use('TKAgg')
    import matplotlib.pyplot as plt
    nx.draw(graph)
    plt.show()


def convert_to_adj_matrix(nx_G, is_directed):
    '''
    Converts a networkx graph object to numpy adjacency matrix
    # Arguments
        nx_G: networkx graph object
        is_directed: Flag to indicate if graph is directed
    # Returns
        Adjacency Matrix Representation
    '''
    num_nodes = nx_G.order()
    adj = np.zeros((num_nodes, num_nodes), dtype = np.int64)
    for edge in nx_G.edges():
        adj[edge[0]][edge[1]] = 1
        if not is_directed:
            adj[edge[1]][edge[0]] = 1

    return adj


def preprocess_input_graph(A, r, T):
    '''
    Preprocess the adjacency matrix according to the formula
    # Arguments
        A: Adjacency Matrix of the input graph
    # Returns
        Preprocessed matrix according to the formula
    '''
    print("Pre-processing input graph to obtain Xs...")

    num_nodes = len(A)
    P0 = np.eye(num_nodes)
    state_trnsn_mat = np.divide(A, A.sum(1)).transpose()

    restart_prob = np.multiply(1 - r, P0)
    P = [P0]
    for i in range(T):
        P_temp = np.add(np.multiply(r, np.matmul(P[i], state_trnsn_mat)), restart_prob)
        P.append(P_temp)

    X = np.average(P[1:], axis = 0)
    return X


def leakyReLU(x, alpha = 0.2):
    '''
    Computes LeakyReLU of a numpy array
    # Arguments
        x: Numpy array
        alpha: Negative slope coefficient
    # Returns
        Numpy array
    '''
    x_np_array = np.asarray(x)

    return np.where(x_np_array > 0, x_np_array, np.multiply(x_np_array, alpha))


def softmax(x):
    '''
    Computes Softmax of a numpy array
    # Arguments
        x: Numpy array
    # Returns
        Numpy array
    '''
    return (np.exp(x).transpose() / np.sum(np.exp(x), axis = 1)).transpose()


def compute_dense(X, A, kernel, attention_kernel, bias):
    '''
    Computes the alpha embeddings for a pair of nodes with an edge between them
    # Arguments
        X: Feature Matrix
        A: Adjacency Matrix
        kernel: Weight Matrix of the layer
        attention_kernel: Weight Matrix of the attention kernel layer
    # Returns
        Dense matrix corresponding to importance of a node on its neighbours
    '''
    features = np.dot(X, kernel)  # (N x F')

    #Compute feature combinations
    # Note: [[a_1], [a_2]]^T [[Wh_i], [Wh_2]] = [a_1]^T [Wh_i] + [a_2]^T [Wh_j]
    attn_for_self = np.dot(features, attention_kernel[0])    # (N x 1), [a_1]^T [Wh_i]
    # attn_for_self = K.print_tensor(attn_for_self, message = "attn_for_self is: ")
    attn_for_neighs = np.dot(features, attention_kernel[1])  # (N x 1), [a_2]^T [Wh_j]
    # attn_for_neighs = K.print_tensor(attn_for_neighs, message = "attn_for_neighs is: ")

    # Attention head a(Wh_i, Wh_j) = a^T [[Wh_i], [Wh_j]]
    dense = attn_for_self + attn_for_neighs.transpose()  # (N x N) via broadcasting

    # Add nonlinearty
    dense = leakyReLU(dense)

    # Mask values before activation (Vaswani et al., 2017)
    mask = -10e9 * (1.0 - A)
    dense += mask

    # Apply softmax to get attention coefficients
    dense = softmax(dense)  # (N x N)

    embeddings = np.dot(dense, features)

    return dense, embeddings
