import numpy as np


def is_same_node(c, t):
    '''
    c start node as shape: [vector length]
    t target node as shape: [vector length] 
    '''
    return np.linalg.norm(c - t) < 1e-4


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Graph:
    def __init__(self, adj_matrix, estimate):
        self.adj_matrix = adj_matrix
        self.estimate = estimate

    def find_path(self, c, t):
        '''
        c start node as shape: [vector length]
        t target node as shape: [vector length]
        '''
        while True:
            if is_same_node(c, t):
                break
            energy = np.matmul(self.adj_matrix, c) + np.matmul(self.estimate, t)
            c = sigmoid(energy)
            yield c


if __name__ == '__main__':
    print("assert that probabilistic graph works.")
    print(resolve_next(np.array([[1, 1], [0, 1]]), np.array([-10, 10])))
