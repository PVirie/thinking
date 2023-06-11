import numpy as np

class Node:
    def __init__(self, data):
        self.data = data
    # is same node
    async def is_same_node(self, another):
        '''
        c start node as shape: [vector length, 1]
        t target node as shape: [vector length, 1]
        '''
        dist = np.linalg.norm(self.data - another.data)
        return dist < 1e-4
