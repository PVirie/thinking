import numpy as np
import asyncio

class Node:
    def __init__(self, data):
        self.data = data
    # is same node
    async def is_same_node(self, another):
        dist = np.linalg.norm(self.data - another.data)
        return dist < 1e-4

    # static method
    @staticmethod
    async def match(node, templates):
        '''
        node.data as shape: [vector length]
        templates as a list of nodes
        '''
        templates = np.array([t.data for t in templates])
        # first flatten shape
        flatten = np.reshape(templates, [-1, templates.shape[-1]])
        results = np.linalg.norm(flatten - node.data, axis=1)
        # then reshape the result back to [...]
        return np.exp(-np.reshape(results, templates.shape[:-1]))
    

async def test():
    node1 = Node(np.array([1, 2, 3]))
    node2 = Node(np.array([1, 2, 3]))
    node3 = Node(np.array([1, 2, 4]))
    print(await node1.is_same_node(node2))
    print(await node1.is_same_node(node3))
    print(await Node.match(node1, [[1, 2, 3], [1, 2, 4]]))


if __name__ == '__main__':
    asyncio.run(test())