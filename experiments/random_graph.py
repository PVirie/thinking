'''
Hypothesis: chance that a node will appear in a shortest path depends on its degree.
'''

import numpy as np
import matplotlib.pyplot as plt
import random


def random_graph(size, p):
    raw = np.random.rand(size, size)
    graph = (raw < p).astype(np.int32)
    graph[np.arange(size), np.arange(size)] = 0
    return graph


def node_degrees(graph):
    out_going = np.sum(graph, axis=1, keepdims=False)
    in_coming = np.sum(graph, axis=0, keepdims=False)
    return out_going, in_coming


def shortest_path(graph, s, d):
    inf = graph.shape[0] * 2
    dist = np.ones([graph.shape[0]]) * inf
    dist[s] = 0
    unvisited = np.ones([graph.shape[0]], dtype=bool)

    indices = np.arange(graph.shape[0])
    trace = np.arange(graph.shape[0], dtype=np.int32)

    c = s
    while True:
        unvisited[c] = False
        uv_neighbor = np.logical_and((graph[c, :] > 0), unvisited).astype(np.float32)
        trace = np.where(dist[c] + graph[c, :] < dist, c, trace)
        dist = dist * (1 - uv_neighbor) + np.minimum(dist, dist[c] + graph[c, :]) * (uv_neighbor)

        c = indices[unvisited][np.argmin(dist[unvisited])]
        if c == d:
            if dist[d] < inf:
                path = []
                while True:
                    path.append(c)
                    if c == s:
                        return path
                    c = trace[c]
            else:
                return []


def random_path(graph, s, d):
    unvisited = np.ones([graph.shape[0]], dtype=bool)
    potential = np.zeros([graph.shape[0]], dtype=bool)

    indices = np.arange(graph.shape[0])
    trace = np.arange(graph.shape[0], dtype=np.int32)

    c = s
    while True:
        unvisited[c] = False
        uv_neighbor = np.logical_and((graph[c, :] > 0), unvisited)
        trace[uv_neighbor] = c
        potential[uv_neighbor] = True
        candidates = indices[np.logical_and(potential, unvisited)]

        if candidates.shape[0] == 0:
            return []
        else:
            c = np.random.choice(candidates, 1)[0]
            if c == d:
                path = []
                while True:
                    path.append(c)
                    if c == s:
                        return path
                    c = trace[c]


def histogram(stats, prop):

    V = {}
    for item in stats:
        if item in V:
            V[item] = V[item] + 1
        else:
            V[item] = 1

    H = {}
    for item in prop:
        if item in H:
            H[item] = H[item] + 1
        else:
            H[item] = 1

    out = []
    for item in H:
        if item not in V:
            out.append((item, 0, 0))
        else:
            out.append((item, V[item], V[item] / H[item]))
    return sorted(out, key=lambda ivvh: ivvh[0])


if __name__ == '__main__':

    prop = []
    stats = []
    for _ in range(10000):
        size = 20
        g = random_graph(size, 0.1)
        o, i = node_degrees(g)
        d = o + i
        prop.extend(d)

        # path = shortest_path(g, 0, size - 1)
        path = random_path(g, 0, size - 1)

        stat = o[path] + i[path]
        stats.extend(stat)

    data = histogram(stats, prop)
    items, Vs, densities = zip(*data)

    plt.bar(items, densities)  # arguments are passed to np.histogram
    plt.plot(items, np.log(items) * 0.1)
    plt.show()
