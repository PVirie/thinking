import jax.numpy as jnp
import asyncio
from typing import List



class Node:
    def __init__(self, data):
        self.data = jnp.array(data)
    # is same node
    async def is_same_node(self, another):
        dist = jnp.linalg.norm(self.data - another.data)
        return dist < 1e-4


class Node_tensor_2D:
    def __init__(self, max_rows, max_cols, data=None, node_dim=16):
        self.max_rows = max_rows
        self.max_cols = max_cols
        if data is not None:
            self.data = jnp.array(data)
            self.node_dim = self.data.shape[-1]
        else:
            self.data = jnp.zeros([max_rows, max_cols, node_dim])
            self.node_dim = node_dim

    async def match(self, node: Node):
        '''
        node.data as shape: [node_dim]
        self.data as shape: [max_rows, max_cols, node_dim]
        '''

        # first flatten template
        flatten = jnp.reshape(self.data, [-1, self.node_dim])
        results = jnp.exp(-jnp.linalg.norm(flatten - node.data, axis=1))

        # need to filter where self.H is invalid (i.e. where the chunk is not full)
        results = jnp.where(jnp.linalg.norm(flatten, axis=1) < 1e-8, 0, results)

        # then reshape the result back to list of list of list
        return jnp.reshape(results, [self.max_rows, self.max_cols])


    async def append(self, hs: List[Node]):
        num_steps = len(hs)

        # these are old obsolute update scheme
        # self.data = np.roll(self.data, -1, axis=0)
        # self.data[self.max_rows - 1, :num_steps, :] = np.stack([h.data for h in hs], axis=0)
        # self.data[self.max_rows - 1, num_steps:, :] = 0
        # use non-assigned update scheme instead

        new_data = jnp.stack([h.data for h in hs], axis=0)

        # Pad new_data to match the size of the last row
        padding = ((0, self.max_cols - num_steps), (0, 0))
        last_row_new_data = jnp.pad(new_data, pad_width=padding, mode='constant', constant_values=0)

        # Replace the last row of the rolled data with the newly created last row
        self.data = jnp.concatenate([self.data[1:], jnp.expand_dims(last_row_new_data, axis=0)], axis=0)



    async def access(self, row, col):
        return Node(self.data[row, col, :])


    async def roll(self, axis, shift):
        new_tensor = Node_tensor_2D(self.max_rows, self.max_cols, jnp.roll(self.data, shift, axis=axis))
        return new_tensor
    

    async def consolidate(self, prop, use_max=False):
        '''
        prop as shape: [max_row, max_cols]
        '''
        # first flatten template
        augmented = jnp.reshape(prop, [-1, 1])
        flatten = jnp.reshape(self.data, [-1, self.node_dim])
        if use_max:
            return Node(flatten[jnp.argmax(augmented), :])
        else:
            return Node(jnp.reshape(jnp.sum(augmented * flatten, axis=0) / jnp.sum(prop), [self.node_dim]))

class Metric_Printer:
    def __init__(self, supports: Node_tensor_2D):
        self.supports = supports

    async def replace(self, x):
        if isinstance(x, Node):
            logits = await self.supports.match(x)
            return jnp.argmax(logits)
        elif isinstance(x, list):
            return [await self.replace(y) for y in x]
        elif isinstance(x, tuple):
            return tuple([await self.replace(y) for y in x])
        elif isinstance(x, dict):
            return {await self.replace(k): await self.replace(v) for k, v in x.items()}
        elif isinstance(x, set):
            return {await self.replace(y) for y in x}
        else:
            return x


    # accept args and kwargs
    async def print(self, *args, **kwargs):
        # replace args that are Node with the corresponding max support's logit
        new_args = []
        for arg in args:
            new_args.append(await self.replace(arg))
            
        print(*new_args, **kwargs)
        



async def test():
    node1 = Node(jnp.array([1, 2, 3]))
    node2 = Node(jnp.array([1, 2, 3]))
    node3 = Node(jnp.array([1, 2, 4]))
    print(await node1.is_same_node(node2))
    print(await node1.is_same_node(node3))

    nodes = Node_tensor_2D(2, 2, node_dim=3)
    await nodes.append([node2, node3])
    await nodes.append(jnp.array([[0, 0, 0], [1, 2, 3]]))
    print(await nodes.match(node1))

    rolled = await nodes.roll(0, 1)
    print(await rolled.match(node1))

    new_node = await nodes.consolidate(jnp.array([[1, 0], [0, 1]]), use_max=False)
    print(new_node.data)
    assert(await new_node.is_same_node(node1))


if __name__ == '__main__':
    asyncio.run(test())