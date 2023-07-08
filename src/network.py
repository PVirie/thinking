import numpy as np
import os
import asyncio
from typing import List
from metric import Node
from loguru import logger


def compute_pivot_indices(entropies):
    all_pvs = []
    for j in range(0, entropies.shape[0] - 1):
        if j > 0 and entropies[j] < entropies[j - 1]:
            all_pvs.append(j - 1)
    all_pvs.append(entropies.shape[0] - 1)
    return all_pvs


class Layer:
    def __init__(self, heuristics, hippocampus, proxy):
        self.heuristics = heuristics
        self.hippocampus = hippocampus
        self.proxy = proxy
        self.next = None

    def assign_next(self, next_layer):
        self.next = next_layer

    async def incrementally_learn(self, path: List[Node]):
        if len(path) < 2:
            return

        await self.hippocampus.incrementally_learn(path)
        await self.proxy.incrementally_learn(path)

        entropies = await self.hippocampus.compute_entropy(path)
        pivots_indices = compute_pivot_indices(entropies)
        pivots = path[pivots_indices]

        await self.heuristics.incrementally_learn(path, pivots)

        if self.next is not None and len(pivots) > 1:
            await self.next.incrementally_learn(pivots)

    async def to_next(self, c: Node, forward=True):
        # not just enhance, but select the closest with highest entropy

        entropy = await self.hippocampus.compute_entropy(c)
        for i in range(1000):
            next_c = await self.hippocampus.sample(c, forward)
            next_entropy = await self.hippocampus.compute_entropy(c)

            if entropy >= next_entropy:
                return c
            c = next_c
            entropy = next_entropy

        raise ValueError('Cannot find a top node in time.')

    async def from_next(self, c: Node):
        return c

    async def pincer_inference(self, s: Node, t: Node, pathway_bias=0):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        candidates, props = await self.proxy.get_candidates(s)

        cortex_rep, cortex_prop = await self.heuristics.consolidate(s, candidates, props, t)
        hippocampus_rep, hippocampus_prop = await self.hippocampus.infer(s, t)

        sup_info = [(self.hippocampus.enhance(cortex_rep), cortex_prop), (self.hippocampus.enhance(hippocampus_rep), hippocampus_prop)]

        if pathway_bias < 0:
            return hippocampus_rep, sup_info
        if pathway_bias > 0:
            return cortex_rep, sup_info

        return hippocampus_rep if hippocampus_prop > cortex_prop else cortex_rep, sup_info

    async def find_path(self, c: Node, t: Node, hard_limit=20, pathway_bias=0):
        # pathway_bias < 0 : use hippocampus
        # pathway_bias > 0 : use cortex

        count_steps = 0

        yield False, {
            "layer": self.name,
            "s": c,
            "t": t
        }

        if self.next is not None:
            goals = await self.next.find_path(self.to_next(c, True), self.to_next(t, False))

        yield False, {
            "layer": self.name,
            "selected": c,
            "choices": []
        }

        yield True, c

        while True:
            if await c.is_same_node(t):
                break

            if self.next is not None:
                try:
                    while True:
                        # pop the next goal
                        is_result, result = await anext(goals)
                        if is_result:
                            yield False, result
                            break
                    g = await self.from_next(result)
                except StopIteration:
                    yield False,{
                        "layer": self.name + 1,
                        "selected": t,
                        "choices": []
                    }
                    g = t
            else:
                g = t

            while True:
                if await c.is_same_node(t):
                    # we have reached the target
                    break

                if await c.is_same_node(g):
                    # we have reached the goal
                    c = g
                    break

                count_steps = count_steps + 1
                if count_steps >= hard_limit:
                    yield False, {
                        "layer": self.name,
                        "success": False,
                    }
                    raise RecursionError

                c, supplementary = await self.pincer_inference(c, g, pathway_bias, with_info=True)
                c = await self.hippocampus.enhance(c)  # enhance signal preventing signal degradation

                yield False, {
                    "layer": self.name,
                    "selected": c,
                    "choices": supplementary
                }
                yield True, c

        yield False, {
            "layer": self.name,
            "success": True,
        }

    async def next_step(self, c: Node, t: Node, pathway_bias=0):
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
    from pathways import heuristic, hippocampus, proxy
    from utilities import *

    np.set_printoptions(precision=2)

    graph_shape = 16

    representations = generate_onehot_representation(np.arange(graph_shape), graph_shape)

    config = {
        "world_update_prior": 0.1,
        "layers": [
            {
                "num_dimensions": graph_shape,
                "memory_slots": 64,
                "chunk_size": 16,
                "diminishing_factor": 0.9,
                "heuristic_model_param": {
                    'pre_steps': 4, 'all_pairs': False,
                    'lr': 0.01, 'step_size': 1000, 'weight_decay': 0.99
                }
            },
            {
                "num_dimensions": graph_shape,
                "memory_slots": 64,
                "chunk_size": 12,
                "diminishing_factor": 0.9,
                "heuristic_model_param": {
                    'pre_steps': 4, 'all_pairs': False,
                    'lr': 0.01, 'step_size': 1000, 'weight_decay': 0.99
                }
            },
            {
                "num_dimensions": graph_shape,
                "memory_slots": 64,
                "chunk_size": 8,
                "diminishing_factor": 0.9,
                "heuristic_model_param": {
                    'pre_steps': 4, 'all_pairs': False,
                    'lr': 0.01, 'step_size': 1000, 'weight_decay': 0.99
                }
            }
        ]
    }


    answer = input("Do you want to retrain? (y/n): ").lower().strip()
    train = answer == "y"

    dir_path = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(dir_path, "..", "artifacts", "weights", "network.py.test")

    if train:
        empty_directory(weight_path)

        g = random_graph(graph_shape, 0.4)
        np.save(os.path.join(weight_path, "graph.npy"), g)
        print(g)

        # cognitive_map = build_cognitive_map(config)
        # print(cognitive_map)
    else:
        g = np.load(os.path.join(weight_path, "graph.npy"))
        print(g)

        # cognitive_map = load_cognitive_map(config, weight_path)
        # print(cognitive_map)


    # with network.build_network(config, weight_path, save_on_exit=train, logger=logger) as root:
    #     print("Training a cognitive map:")
    #     for i in range(explore_steps):
    #         path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
    #         path = all_reps[:, path]
    #         root.incrementally_learn(path)
    #         if i % (explore_steps // 100) == 0:
    #             print("Training progress: %.2f%%" % (i * 100 / explore_steps), end="\r", flush=True)
    #     print("Finish learning.")

    # goals = range(graph_shape)
    # max_steps = 40

    # total_length = 0
    # stamp = time.time()
    # for t in goals:
    #     try:
    #         p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps)
    #         for pi in p:
    #             print(np.argmax(pi), end=' ')
    #             total_length = total_length + 1
    #     except RecursionError:
    #         print("fail to find path in time.", t, end=' ')
    #     finally:
    #         print()
    # print("cognitive planner:", time.time() - stamp, " average length:", total_length / len(goals))

    # total_length = 0
    # stamp = time.time()
    # for t in goals:
    #     try:
    #         p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps, pathway_bias=-1)
    #         for pi in p:
    #             print(np.argmax(pi), end=' ')
    #             total_length = total_length + 1
    #     except RecursionError:
    #         print("fail to find path in time.", t, end=' ')
    #     finally:
    #         print()
    # print("hippocampus planner:", time.time() - stamp, " average length:", total_length / len(goals))

    # total_length = 0
    # stamp = time.time()
    # for t in goals:
    #     try:
    #         p = cognitive_map.find_path(representations[:, 0:1], representations[:, t:(t + 1)], hard_limit=max_steps, pathway_bias=1)
    #         for pi in p:
    #             print(np.argmax(pi), end=' ')
    #             total_length = total_length + 1
    #     except RecursionError:
    #         print("fail to find path in time.", t, end=' ')
    #     finally:
    #         print()
    # print("cortex planner:", time.time() - stamp, " average length:", total_length / len(goals))

    # total_length = 0
    # stamp = time.time()
    # for t in goals:
    #     try:
    #         p = shortest_path(g, 0, t)
    #         p = list(reversed(p))
    #         for pi in p:
    #             print(pi, end=' ')
    #             total_length = total_length + 1
    #     except RecursionError:
    #         print("fail to find path in time.", t, end=' ')
    #     finally:
    #         print()

    # print("optimal planner:", time.time() - stamp, " average length:", total_length / len(goals))