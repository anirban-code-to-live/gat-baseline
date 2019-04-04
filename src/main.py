import numpy as np
import matplotlib
import networkx as nx
from gat.gat import learn_model
import utils.common_utils as common_utils
import utils.graph_utils as graph_utils

import time
import json
import csv


def main(args):
    # read json file for number of class labels
    with open('../data/num_classes.json') as json_data:
        class_json = json.load(json_data)
    dataset = args.dataset
    num_classes = class_json['datasets'][dataset]

    # read the graph as a networkx object
    print('Running Attention2Vec for %s dataset\n' % dataset)
    nx_G, labels = graph_utils.read_graph(dataset, args.weighted, args.directed, num_classes)
    adjacency_matrix = graph_utils.convert_to_adj_matrix(nx_G, args.directed)

    print("Number of nodes in the graph : %s" % len(nx_G.nodes()))
    print("Number of edges in the structure graph : %s" % len(nx_G.edges()))

    # preprocess the input adjacency matrix
    t = time.time()
    X = adjacency_matrix

    # read the training data
    train_per = args.train_per # to be changed
    train_nodes = []
    with open('../data/{}/training/{}_{}_train.csv'.format(dataset, dataset, train_per)) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            train_nodes.append(int(row[0]))
    val_nodes = list(set(range(labels.shape[0])) - set(train_nodes))

    kernel, attention_kernel, bias = learn_model(X, adjacency_matrix, labels, train_nodes, val_nodes)
    print("learn_model is taking " + str(time.time() - t) + "s\n")

    alpha_ij, attn_embeds = graph_utils.compute_dense(X, adjacency_matrix, kernel, attention_kernel, bias)
    print("compute_dense is taking " + str(time.time() - t) + "s\n")
    np.savetxt('../data/' + args.dataset + '/' + args.dataset + '.embed.csv', attn_embeds, fmt='%.6f')


if __name__ == "__main__":
    # parse the arguments
    args = common_utils.parse_arguments()
    # # setup the logger
    # common_utils.set_up_logger()
    main(args)
