import argparse
import logging
import numpy as np


def parse_arguments():
    '''
    Parses the attention2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run attention2vec.")

    parser.add_argument('--train_per', type=int, default=20,
                        help='Input train percentage')

    parser.add_argument('--dataset', nargs='?', default='cora',
                        help='Input graph name for saving files')

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
