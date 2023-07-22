import numpy as np
import asyncio
from typing import List



class Node:
    def __init__(self, data):
        self.data = np.array(data)
    # is same node
    async def is_same_node(self, another):
        dist = np.linalg.norm(self.data - another.data)
        return dist < 1e-4


class Node_tensor_2D:
    def __init__(self, max_rows, max_cols, data=None, node_dim=16):
        self.max_rows = max_rows
        self.max_cols = max_cols
        if data is not None:
            self.data = np.array(data)
            self.node_dim = self.data.shape[-1]
        else:
            self.data = np.zeros([max_rows, max_cols, node_dim])
            self.node_dim = node_dim

    async def match(self, node: Node):
        '''
        node.data as shape: [node_dim]
        self.data as shape: [max_rows, max_cols, node_dim]
        '''

        # first flatten template
        flatten = np.reshape(self.data, [-1, self.node_dim])
        results = np.exp(-np.linalg.norm(flatten - node.data, axis=1))

        # need to filter where self.H is invalid (i.e. where the chunk is not full)
        results = np.where(np.linalg.norm(flatten, axis=1) < 1e-8, 0, results)

        # then reshape the result back to list of list of list
        return np.reshape(results, [self.max_rows, self.max_cols])


    async def append(self, hs: List[Node]):
        num_steps = len(hs)
        self.data = np.roll(self.data, -1, axis=0)
        self.data[self.max_rows - 1, :num_steps, :] = np.array([h.data for h in hs])
        self.data[self.max_rows - 1, num_steps:, :] = 0


    async def access(self, row, col):
        return Node(self.data[row, col, :])


    async def roll(self, axis, shift):
        new_tensor = Node_tensor_2D(self.max_rows, self.max_cols, np.roll(self.data, shift, axis=axis))
        return new_tensor
    

    async def consolidate(self, prop, use_max=False):
        '''
        prop as shape: [max_row, max_cols]
        '''
        # first flatten template
        augmented = np.reshape(prop, [-1, 1])
        flatten = np.reshape(self.data, [-1, self.node_dim])
        # then reshape the result back to list of list of list
        if use_max:
            return Node(flatten[np.argmax(augmented), :])
        else:
            return Node(np.reshape(np.sum(augmented * flatten, axis=0) / np.sum(prop), [self.node_dim]))


async def test():
    node1 = Node(np.array([1, 2, 3]))
    node2 = Node(np.array([1, 2, 3]))
    node3 = Node(np.array([1, 2, 4]))
    print(await node1.is_same_node(node2))
    print(await node1.is_same_node(node3))

    nodes = Node_tensor_2D(2, 2, node_dim=3)
    await nodes.append([node2, node3])
    await nodes.append(np.array([[0, 0, 0], [1, 2, 3]]))
    print(await nodes.match(node1))

    rolled = await nodes.roll(0, 1)
    print(await rolled.match(node1))

    new_node = await nodes.consolidate(np.array([[1, 0], [0, 1]]), use_max=False)
    print(new_node.data)
    assert(await new_node.is_same_node(node1))


if __name__ == '__main__':
    asyncio.run(test())