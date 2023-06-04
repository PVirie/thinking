import numpy as np
from contextlib import contextmanager
import heuristic
import hippocampus
import os
import asyncio
from loguru import logger


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


def compute_pivot_indices(entropies):
    all_pvs = []
    for j in range(0, entropies.shape[0] - 1):
        if j > 0 and entropies[j] < entropies[j - 1]:
            all_pvs.append(j - 1)
    all_pvs.append(entropies.shape[0] - 1)
    return all_pvs

class Layer:
    def __init__(self, heuristics, hippocampus):
        self.heuristics = heuristics
        self.hippocampus = hippocampus
        self.next = None

    def assign_next(self, next_layer):
        self.next = next_layer

    async def incrementally_learn(self, path):
        if len(path) < 2:
            return

        await self.hippocampus.incrementally_learn(path)

        entropies = await self.hippocampus.compute_entropy(path)
        pivots_indices = compute_pivot_indices(entropies)
        pivots = path[pivots_indices]

        await self.heuristics.incrementally_learn(path, pivots)

        if self.next is not None and len(pivots) > 1:
            await self.next.incrementally_learn(pivots)

    async def to_next(self, c, forward=True):
        # not just enhance, but select the closest with highest entropy
        # c has shape [dimensions, 1]

        entropy = await self.hippocampus.compute_entropy(c)
        for i in range(1000):
            C, c_prop = await self.hippocampus.get_distinct_next_candidate(c, forward)
            ent_scores = await self.hippocampus.compute_entropy(C)

            scores = c_prop * ent_scores
            next_index = np.argmax(scores, axis=0)
            next_c = C[next_index]
            next_entropy = scores[next_index]

            # alternative sum method for generalization
            # weights = np.where(scores > entropy, 1.0, 0.0)
            # next_c = np.sum(C * np.reshape(weights, [self.num_dimensions, -1]), axis=1, keepdims=True)
            # next_c = self.hippocampus.enhance(next_c)
            # next_entropy = self.hippocampus.compute_entropy(next_c)

            if entropy >= next_entropy:
                return c
            c = next_c
            entropy = next_entropy

        raise ValueError('Cannot find a top node in time.')

    async def from_next(self, c):
        return c

    async def pincer_inference(self, s, t, pathway_bias=0):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        candidates, props = await self.hippocampus.get_distinct_next_candidate(s)

        cortex_rep, cortex_prop = await self.heuristics.consolidate(s, candidates, props, t)
        hippocampus_rep, hippocampus_prop = await self.hippocampus.infer(s, t)

        # info = [(self.hippocampus.enhance(cortex_rep), cortex_prop), (self.hippocampus.enhance(hippocampus_rep), hippocampus_prop)]

        if pathway_bias < 0:
            return hippocampus_rep
        if pathway_bias > 0:
            return cortex_rep

        return hippocampus_rep if hippocampus_prop > cortex_prop else cortex_rep

    async def find_path(self, c, t, hard_limit=20, pathway_bias=0):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        count_steps = 0

        logger.info({
            "layer": self.name,
            "s": c,
            "t": t
        })

        if self.next is not None:
            goals = await self.next.find_path(self.to_next(c, True), self.to_next(t, False))

        logger.info({
            "layer": self.name,
            "selected": c,
            "choices": []
        })

        yield c

        while True:
            if await c.is_same_node(t):
                break

            if self.next is not None:
                try:
                    g = await self.from_next(await anext(goals))
                except StopIteration:
                    logger.info({
                        "layer": self.name + 1,
                        "selected": t,
                        "choices": []
                    })
                    g = t
            else:
                g = t

            while True:
                if await c.is_same_node(t):
                    # wait we found the true target!
                    break

                if await c.is_same_node(g):
                    c = g
                    break

                count_steps = count_steps + 1
                if count_steps >= hard_limit:
                    logger.info({
                        "layer": self.name,
                        "success": False,
                    })
                    raise RecursionError
                    break

                c, supplementary = await self.pincer_inference(c, g, pathway_bias, with_info=True)
                c = await self.hippocampus.enhance(c)  # enhance signal preventing signal degradation

                logger.info({
                    "layer": self.name,
                    "selected": c,
                    "choices": [(self.hippocampus.enhance(r), p) for r, p in supplementary]
                })
                yield c

        logger.info({
            "layer": self.name,
            "success": True,
        })

    async def next_step(self, c, t, pathway_bias=0):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        if await c.is_same_node(t):
            return t

        next_g = await self.next.next_step(await self.to_next(c), await self.to_next(t))
        g = await self.from_next(next_g)

        if await c.is_same_node(g):
            return g

        c, supplementary = await self.pincer_inference(c, g, pathway_bias)
        return c


if __name__ == '__main__':
    print("assert that probabilistic network works.")
