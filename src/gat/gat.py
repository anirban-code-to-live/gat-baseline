from __future__ import division, print_function

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
from keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from keras.layers import Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

from gat.graph_attention_layer import GraphAttention
from utils.graph_utils import compute_dense
from utils.common_utils import sample_mask

# set logging only for Error
tf.logging.set_verbosity(tf.logging.ERROR)

def learn_model(X, adj_matrix, labels, train_nodes, val_nodes):
    '''
    Create a GAT Layer to learn the Kernel and Attention kernel
    Used to compute the alpha matrix using a single forward pass
    # Arguments
        X: The feature Matrix
        adjacency_matrix: Adjacency Matrix of the graph
        labels: Labels of each node (0-6)
    # Returns
        The learnt kernel and attention_kernel numpy arrays
    '''
    # Parameters
    N = X.shape[0]                # Number of nodes in the graph
    Y_train, Y_val, Y_test, idx_train, idx_val, idx_test = generate_train_val_test_data(labels, train_nodes, val_nodes)

    F = X.shape[1]                # Original feature dimension
    n_classes = Y_train.shape[1]  # Number of classes
    F_ = 128                      # Output size of GraphAttention layer
    n_attn_heads = 1              # Number of attention heads in GAT layer
    dropout_rate = 0.6            # Dropout rate (between and inside GAT layers)
    l2_reg = 5e-4/2               # Factor for l2 regularization
    learning_rate = 5e-3          # Learning rate for Adam
    epochs = 500                # Number of training epochs
    es_patience = 100             # Patience for early stopping

    # Add self-loops
    adjacency_matrix = np.array(adj_matrix)
    for i in range(adjacency_matrix.shape[0]):
        adjacency_matrix[i][i] = 1

    # Model definition (as per Section 3.3 of the paper)
    X_in = Input(shape = (F, ))
    A_in = Input(shape = (N, ))

    dropout = Dropout(dropout_rate)(X_in)
    graph_attention_1 = GraphAttention(n_classes,
                                       name="graph_layer_1",
                                       attn_heads = n_attn_heads,
                                       attn_heads_reduction = 'average',
                                       dropout_rate = dropout_rate,
                                       activation = 'softmax',
                                       kernel_regularizer = l2(l2_reg),
                                       attn_kernel_regularizer = l2(l2_reg))([dropout, A_in])

    # dropout2 = Dropout(dropout_rate)(graph_attention_1)
    # graph_attention_2 = GraphAttention(n_classes,
    #                                    name="graph_layer_2",
    #                                    attn_heads=1,
    #                                    attn_heads_reduction='average',
    #                                    dropout_rate=dropout_rate,
    #                                    activation='softmax',
    #                                    kernel_regularizer=l2(l2_reg),
    #                                    attn_kernel_regularizer=l2(l2_reg))([dropout2, A_in])

    # Build model
    model = Model(inputs = [X_in, A_in], outputs = graph_attention_1)
    optimizer = Adam(lr = learning_rate)
    model.compile(optimizer = optimizer,
                  loss = 'categorical_crossentropy',
                  weighted_metrics = ['acc'])

    # Callbacks
    es_callback = EarlyStopping(monitor = 'val_weighted_acc', patience = es_patience)
    tb_callback = TensorBoard(batch_size = N)
    mc_callback = ModelCheckpoint('logs/best_model.h5',
                                  monitor = 'val_weighted_acc',
                                  save_best_only = True,
                                  save_weights_only = True)

    # Train model
    validation_data = ([X, adjacency_matrix], Y_val, idx_val)
    model.fit([X, adjacency_matrix],
              Y_train,
              sample_weight = idx_train,
              epochs = epochs,
              batch_size = N,
              validation_data = validation_data,
              shuffle = True,  # Shuffling data means shuffling the whole graph
              callbacks = [es_callback, tb_callback, mc_callback])

    for layer in model.layers:
        if (isinstance(layer, GraphAttention) == True):
            parameters = layer.get_weights()

    # kernel = parameters[0]
    # attention_kernel = (parameters[2], parameters[3])

    parameters_layer_1 = model.get_layer("graph_layer_1").get_weights()
    # parameters_layer_2 = model.get_layer("graph_layer_2").get_weights()

    kernel_1 = parameters_layer_1[0]
    bias_1 = parameters_layer_1[1]
    attention_kernel_1 = (parameters_layer_1[2], parameters_layer_1[3])

    # kernel_2 = parameters_layer_2[0]
    # bias_2 = parameters_layer_2[1]
    # attention_kernel_2 = (parameters_layer_2[2], parameters_layer_2[3])

    return kernel_1, attention_kernel_1, bias_1


def generate_train_val_test_data(labels, train_nodes, val_nodes):
    '''
    Generates the training, validation and testing data from labels
    # Arguments
        labels: Labels corresponding to each node
    # Returns y_train, y_val, y_test, train_mask, val_mask, test_mask
    '''

    train_mask = sample_mask(train_nodes, labels.shape[0])
    val_mask = sample_mask(val_nodes, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)

    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]

    return y_train, y_val, None, train_mask, val_mask, None
