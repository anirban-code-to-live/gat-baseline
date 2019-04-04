import numpy as np
import networkx as nx
import random


class Attention2Vec():
    def __init__(self, nx_G, is_directed, is_weighted):
        self.graph = nx_G
        self.is_directed = is_directed
        self.is_weighted = is_weighted


    def init_embedding_parameters(self, walk_length, num_walks):
        self.walk_length = walk_length
        self.num_walks = num_walks


    def simulate_walks(self):
        '''
        Repeatedly simulate random walks from each node
        # Arguments
            num_walks: Number of random walks from each node
            walk_length: Length of the walk generated from each node
        # Returns
            List of walks generated
        '''
        graph = self.graph
        walks = []
        nodes = list(graph.nodes())
        for walk_iter in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.get_walk_from_node(node))

        return walks


    def get_walk_from_node(self, start_node):
        '''
        Simulate a random walk starting from start node
        # Arguments
            start_node: The starting node of every walk generated
        # Returns
            A walk generated from start_node
        '''
        walk = [start_node]
        for walk_iter in range(self.walk_length):
            current_node = walk[-1]
            sampled_node = self.get_sampled_node(self.alias_prob[current_node][0], self.alias_prob[current_node][1])
            walk.append(sampled_node)

        return walk


    def alias_setup(self, probabilities, nbrs):
        '''
        Compute the alias method of given probabilities
        # Arguments
            probabilities: List of probabilities
            nbrs: Node numbers corresponding to each probability
        # Returns
            A tuple of probability and its alias node
        '''
        small = []
        large = []

        size = len(probabilities)

        prob = [(0, 0.)]*size
        alias = [0]*size

        for index, probability in enumerate(probabilities):
            prob[index] = (nbrs[index], size * probability)
            if (prob[index][1] < 1.0):
                small.append(index)
            else:
                large.append(index)

        while (len(small) > 0 and len(large) > 0):
            s = small.pop()
            l = large.pop()
            alias[s] = prob[l][0]
            prob[l] = (prob[l][0], prob[l][1] + prob[s][1] - 1.0)
            if (prob[l][1] < 1.0):
                small.append(l)
            else:
                large.append(l)

        return prob, alias


    def get_sampled_node(self, prob, alias):
        '''
        Sample a node in a typical 2-D Darts Game Fashion
        # Arguments
            prob: Probability
            alias: Its alias
        # Returns
            Returns a sampled node after the random process
        '''
        num = random.randint(0, len(prob) - 1)

        if random.random() < prob[num][1]:
            return prob[num][0]
        else:
            return alias[num]


    def preprocess_transition_probs(self, alpha_ij):
        '''
        Preprocessing of transition probabilities for guiding the random walks
        # Arguments
            alpha_ij: Dense Matrix computed using GAT
        # Returns
            Generates the transition probabilities
        '''
        alias_prob = {}
        for i, node in enumerate(alpha_ij):
            nbrs = sorted(self.graph.neighbors(i))
            probabilities = [prob for prob in node if prob > 0]
            alias_prob[i] = self.alias_setup(probabilities, nbrs)

        self.alias_prob = alias_prob
