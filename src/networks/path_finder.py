import numpy as np


def shortest_path(energy_model, comparison, s, d):
    inf = graph.shape[0] * 2
    dist = np.ones([graph.shape[0]]) * inf
    dist[s] = 0
    unvisited = np.ones([graph.shape[0]], dtype=np.bool)

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
