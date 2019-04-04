import argparse
import logging
import numpy as np


def parse_arguments():
    '''
    Parses the attention2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run attention2vec.")

    parser.add_argument('--input', nargs='?', default='../data/cora/cora.edgelist',
                        help='Input graph path')

    parser.add_argument('--train_per', type=int, default=20,
                        help='Input train percentage')

    parser.add_argument('--dataset', nargs='?', default='cora',
                        help='Input graph name for saving files')

    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk_length', type=int, default=80,
                        help='Length of walk per source. Default is 80.')

    parser.add_argument('--num_walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')

    parser.add_argument('--window_size', type=int, default=10,
                        help='Context size for optimization. Default is 10.')

    parser.add_argument('--iter', default=3, type=int,
                        help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted Structure-Layer. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')

    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Structure-Layer Graph is (un)directed. Default is undirected.')

    parser.add_argument('--undirected', dest='undirected', action='store_false')

    parser.set_defaults(directed=False)

    return parser.parse_args()


def set_up_logger(mode = 'w'):
    '''
    Sets up the logger for the attention2vec
    '''
    logging.basicConfig(format='%(levelname)s:%(message)s', filename = '../logs/attention2Vec.log',
                        filemode = mode, level = logging.DEBUG)

    logging.debug("Starting Execution...\n\n")


def sample_mask(idx, list, type=np.bool, default_mask=1):
    '''
    Create mask of input vector
    # Arguments
        idx: Index to be set true/1
        list: Length of the vector
        type: Data type of the list
    # Returns
        Masked list
    '''
    mask = np.zeros(list)
    mask[idx] = default_mask

    return np.array(mask, dtype=type)
