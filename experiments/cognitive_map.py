'''
Hypothesis: cognitive maps allow the discovery of competitive min-cost paths.
'''
import numpy as np
from random_graph import *
import random


def build_estimate_and_seed_graph(graph, seed):
    pass


def build_layer_basic(graph):
    encoder = np.arange(graph.shape[0])
    decoder = np.arange(graph.shape[0])
    graph = graph
    top, estimate = build_layer_strategy_A(graph)

    return Layer(encoder, decoder, graph, estimate, top), None


def build_layer_strategy_A(graph):
	print("building another layer with strategy A.")
	
    out_going, in_coming = node_degrees(graph)
    degrees = out_going + in_coming
    pivots = np.arange(graph.shape[0])[degrees > np.mean(degrees)/2]
    decoder = pivots

    # initialize pivot graph
    pivot_graph = np.zeros([pivots.shape[0], pivots.shape[0]], dtype=np.float32)

    # initialize encoder
    encoder = np.ones(graph.shape[0]) * -1
    encoder[pivots] = np.arange(pivots.shape[0]) # encoder is not complete 

    # initialize estimand
    inf = graph.shape[0] * 2
    estimand = np.ones_like(graph) * inf

    step = 1000
    step_size = inf/1000
    trace = []
    for i in range(step):
    	path = random_path(graph, random.randint(graph.shape[0]), graph.shape[0])
    	pv_index = None
    	for v in path:
    		estimand[trace , v] = estimand[trace , v] - step_size    		
    		# check if one of the pivots
    		v_index = encoder[v]
    		if v_index >= 0:
    			# link pivot graph
    			if pv_index is not None:
    				pivot_graph[pv_index , v_index] = 1 # this is for 0, 1 graph only
    			pv_index = v_index
    			# also if we found a pivot, clear trace
	    		trace.clear()

    		trace.append(v)

    est_pivots = estimand[pivots, :] + np.transpose(estimand)[pivots, :] 
	encoder = np.argmin(est_pivots, axis=0)

    top, estimate = build_layer_strategy_A(pivot_graph)
    return Layer(encoder, decoder, pivot_graph, estimate, top), estimand

class Layer:

    def __init__(self, encoder, decoder, graph, estimate, top):
        self.encoder = encoder
        self.decoder = decoder
        self.graph = graph
        self.estimate = estimate
        self.top = top
        self.indices = np.arange(self.graph.shape[0])

    def encode(self, s):
        return self.encoder[s]

    def decode(self, s):
        return self.decoder[s]

    def get_all_vertices(self):
        return self.indices

    def __find_path(self, c, t):

        unvisited = np.ones([self.graph.shape[0]], dtype=np.bool)
        while True:
            if c == t:
                break
            unvisited[c] = False
            neighbors = np.logical_and((self.graph[c, :] > 0), unvisited).astype(np.float32)
            c = self.indices[neighbors][np.argmin(self.graph[c, neighbors] + self.estimate[neighbors, t])]
            yield c

    def find_path(self, c, t):
        if self.top is not None:
            goals = self.top.find_path(self.top.encode(c), self.top.encode(t))

        while True:
            if c == t:
                break

            if self.top is not None:
                try:
                    g = self.top.decode(next(goals))
                except StopIteration:
                    g = t
            else:
                g = t

            path = self.__find_path(c, g)
            for c_ in path:
                yield c_
            c = g


class Cognitive_map:

    def __init__(self):
        pass

    def __build_hierarchy(self, graph):
        self.base_hierarchy, _ = build_layer_basic(graph)

    def find_path(self, s, t):
        path = self.base_hierarchy.find_path(s, t)
        print(list(path))


if __name__ == '__main__':
    # cognitive_map = Cognitive_map()

    # g = random_graph(20, 0.3)
    print(np.where(np.arange(10) == 9))