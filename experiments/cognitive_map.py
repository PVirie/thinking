'''
Hypothesis: cognitive maps allow the discovery of competitive min-cost paths.
'''
import numpy as np
from random_graph import *
import random
import time


def random_walk(graph, s, max_steps):
    unvisited = np.ones([graph.shape[0]], dtype=np.bool)
    indices = np.arange(graph.shape[0])
    path = []
    c = s
    for i in range(max_steps):
        path.append(c)
        unvisited[c] = False
        candidates = indices[np.logical_and((graph[c, :] > 0), unvisited)]
        if candidates.shape[0] == 0:
            return path
        candidate_weights = 1.0 / graph[c, candidates]  # simple PDF
        c = np.random.choice(candidates, 1, p=candidate_weights / np.sum(candidate_weights))[0]
    return path


def build_estimate_and_seed_graph(graph, seed):
    pass


def build_layer_basic(graph):
    encoder = np.arange(graph.shape[0], dtype=np.int32)
    decoder = np.arange(graph.shape[0], dtype=np.int32)
    top, estimate = build_layer_strategy_A(graph)

    return Layer(encoder, encoder, decoder, Graph(graph, estimate), top), None


def build_layer_strategy_A(graph, explore_steps=2000):

    out_going, in_coming = node_degrees(graph)
    degrees = out_going + in_coming
    pivots = np.arange(graph.shape[0])[degrees > np.mean(degrees)]
    print("building another layer with strategy A:", pivots.shape[0])

    if pivots.shape[0] == 0:
        # initialize estimand
        inf = graph.shape[0] * 2
        estimand = np.ones_like(graph) * inf
        np.fill_diagonal(estimand, 0)

        for i in range(explore_steps):
            path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
            last_v = None
            trace = []
            trace_cost = []
            for v in path:
                if len(trace) > 0:
                    new_cost = graph[last_v, v]
                    trace_cost = [trace_cost[k] + new_cost for k in range(len(trace_cost))]
                    estimand[trace, v] = np.minimum(estimand[trace, v], np.array(trace_cost))

                trace.append(v)
                trace_cost.append(0)
                last_v = v

        return None, estimand

    decoder = pivots

    # initialize pivot graph
    pivot_graph = np.zeros([pivots.shape[0], pivots.shape[0]], dtype=np.float32)

    # initialize encoder
    encoder = {}
    for i, p in enumerate(pivots):
        encoder[p] = i

    # initialize estimand
    inf = graph.shape[0] * 2
    estimand = np.ones_like(graph) * inf
    np.fill_diagonal(estimand, 0)

    for i in range(explore_steps):
        path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
        pv_index = None
        last_v = None
        trace = []
        trace_cost = []
        for v in path:
            if len(trace) > 0:
                new_cost = graph[last_v, v]
                trace_cost = [trace_cost[k] + new_cost for k in range(len(trace_cost))]
                estimand[trace, v] = np.minimum(estimand[trace, v], np.array(trace_cost))
            # check if one of the pivots
            if v in encoder:
                v_index = encoder[v]
                # link pivot graph
                if pv_index is not None:
                    pivot_graph[pv_index, v_index] = 1  # this is for 0, 1 graph only
                pv_index = v_index
                # also if we found a pivot, clear trace
                trace.clear()
                trace_cost.clear()

            trace.append(v)
            trace_cost.append(0)
            last_v = v

    forward_encoder = np.argmin(estimand[:, pivots], axis=1)
    backward_encoder = np.argmin(estimand[pivots, :], axis=0)

    top, estimate = build_layer_strategy_A(pivot_graph)
    return Layer(forward_encoder, backward_encoder, decoder, Graph(pivot_graph, estimate), top), estimand


class Graph:
    def __init__(self, adj_matrix, estimate):
        self.adj_matrix = adj_matrix
        self.estimate = estimate
        self.indices = np.arange(self.adj_matrix.shape[0], dtype=np.int32)

    def find_path(self, c, t):

        unvisited = np.ones([self.adj_matrix.shape[0]], dtype=np.bool)
        while True:
            if c == t:
                break
            unvisited[c] = False
            neighbors = np.logical_and((self.adj_matrix[c, :] > 0), unvisited)
            c = self.indices[neighbors][np.argmin(self.adj_matrix[c, neighbors] + self.estimate[neighbors, t])]
            yield c

class Layer:

    def __init__(self, forward_encoder, backward_encoder, decoder, graph, top):
        self.forward_encoder = forward_encoder
        self.backward_encoder = backward_encoder
        self.decoder = decoder
        self.graph = graph
        self.top = top

    def looking_forward_encode(self, s):
        return self.forward_encoder[s]

    def looking_backward_encode(self, s):
        return self.backward_encoder[s]

    def decode(self, s):
        return self.decoder[s]

    def find_path(self, c, t):
        if self.top is not None:
            goals = self.top.find_path(self.top.looking_forward_encode(c), self.top.looking_backward_encode(t))

        yield c
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

            path = self.graph.find_path(c, g)
            for c_ in path:
                yield c_
            c = g


class Cognitive_map:

    def __init__(self):
        pass

    def build_hierarchy(self, graph):
        self.base_hierarchy, _ = build_layer_basic(graph)

    def find_path(self, s, t):
        path = self.base_hierarchy.find_path(s, t)
        return list(path)


if __name__ == '__main__':

    g = random_graph(256, 0.2)
    cognitive_map = Cognitive_map()
    cognitive_map.build_hierarchy(g)

    goals = random.sample(range(g.shape[0]), g.shape[0] // 4)

    total_length = 0
    stamp = time.time()
    for t in goals:
        p = cognitive_map.find_path(0, t)
        total_length = total_length + len(p)
        print(p)
    print("hierarchy planner:", time.time() - stamp, " average length:", total_length / len(goals))

    total_length = 0
    stamp = time.time()
    for t in goals:
        p = list(reversed(shortest_path(g, 0, t)))
        total_length = total_length + len(p)
        print(p)
    print("optimal planner:", time.time() - stamp, " average length:", total_length / len(goals))

    total_length = 0
    stamp = time.time()
    for t in goals:
        p = list(reversed(random_path(g, 0, t)))
        total_length = total_length + len(p)
        print(p)
    print("random planner:", time.time() - stamp, " average length:", total_length / len(goals))
