import time
import os
import shutil
import numpy as np


def empty_directory(output_dir):
    print("Clearing directory: {}".format(output_dir))
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            os.remove(os.path.join(root, file))
        for directory in dirs:
            shutil.rmtree(os.path.join(root, directory))


class Timer:
    def __init__(self, msg):
        self.msg = msg
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.msg % (time.time() - self.start_time))


def generate_onehot_representation(d, max_digits=8):
    b = np.zeros((d.size, max_digits), dtype=np.float32)
    b[np.arange(d.size), d] = 1
    return b


def random_graph(size, p):
    raw = np.random.rand(size, size)
    graph = (raw < p).astype(np.int32)
    graph[np.arange(size), np.arange(size)] = 0
    return graph


def random_walk(graph, s, max_steps):
    # row index = from, column index = to
    unvisited = np.ones([graph.shape[0]], dtype=bool)
    indices = np.arange(graph.shape[0])
    path = []
    c = s
    for i in range(max_steps):
        path.append(c)
        unvisited[c] = False
        candidates = indices[np.logical_and((graph[c, :] > 0), unvisited)]
        if candidates.shape[0] == 0:
            return path
        candidate_weights = np.exp(-graph[c, candidates])  # graph contains costs
        candidate_weights = candidate_weights / np.sum(candidate_weights)
        c = np.random.choice(candidates, 1, p=candidate_weights)[0]
    return path


def shortest_path(graph, s, d):
    if s == d:
        return [s]

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
        trace = np.where(np.logical_and((graph[c, :] > 0), dist[c] + graph[c, :] < dist), c, trace)
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


def max_match(x, H):
    H_ = np.transpose(H)

    p = (np.linalg.norm(H_, ord=1, axis=1, keepdims=True) > 1e-4).astype(np.float32)
    q = (np.linalg.norm(x, ord=1, axis=0, keepdims=True) > 1e-4).astype(np.float32)

    H_ = np.argmax(H_, axis=1, keepdims=True)
    x = np.argmax(x, axis=0, keepdims=True)

    prop = (H_ == x).astype(np.float32) * p * q

    return prop


if __name__ == '__main__':
    graph_shape = 16
    g = random_graph(graph_shape, 0.2)
    print(g)
    for i in range(10):
        path = random_walk(g, 0, 10)
        print(path)
