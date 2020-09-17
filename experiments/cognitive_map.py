'''
Hypothesis: cognitive maps allow the discovery of competitive min-cost paths.
'''
import numpy as np
from random_graph import *


def propagate(graph):
    for j in range(2):
        print(">", graph)
        temp = []
        for i in range(graph.shape[0]):
            r = np.copy(graph[i, :])
            r[i] = 0
            h = np.copy(graph[i, :])
            h[i] = 1
            temp.append(np.amin((np.reshape(r, [-1, 1]) + graph)[h > 0, :], axis=0))
        graph = np.stack(temp, axis=0)
    print(">", graph)


def build_layer_basic(graph):
    encoder = np.arange(graph.shape[0])
    decoder = np.arange(graph.shape[0])
    graph = graph
    top = build_layer_strategy_A(graph)
    pivots = top.decode(top.get_all_vertices())

    estimate = graph
    # p = self.indices[np.argmin(self.estimate[self.map[s], :])]

    return Layer(encoder, decoder, graph, estimate, top)


def build_layer_strategy_A(graph):

    out_going, in_coming = node_degrees(graph)
    degrees = out_going + in_coming

    encoder = np.arange(graph.shape[0])
    decoder = np.arange(graph.shape[0])
    graph = graph
    top = Layer(graph)
    pivots = top.decode(top.get_all_vertices())
    estimate = graph

    return Layer(encoder, decoder, graph, estimate, top)

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
        self.base_hierarchy = build_layer_basic(graph)

    def find_path(self, s, t):
        path = self.base_hierarchy.find_path(s, t)
        print(list(path))


if __name__ == '__main__':
    # cognitive_map = Cognitive_map()

    g = random_graph(4, 0.3)
    propagate(g)
