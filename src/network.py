import numpy as np
import os
import asyncio
from typing import List
from metric import Node, Node_tensor_2D, Metric_Printer
from loguru import logger
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--retrain", action="store_true")
args = parser.parse_args()


def compute_pivot_indices(entropies):
    all_pvs = []
    for j in range(0, len(entropies) - 1):
        if j > 0 and entropies[j] < entropies[j - 1]:
            all_pvs.append(j - 1)
    all_pvs.append(len(entropies) - 1)
    return all_pvs



class Layer:
    def __init__(self, name, heuristics, hippocampus, proxy):
        self.name = name
        self.heuristics = heuristics
        self.hippocampus = hippocampus
        self.proxy = proxy
        self.next = None

    def assign_next(self, next_layer):
        self.next = next_layer

    def save(self, weight_path, index=0):
        layer_path = os.path.join(weight_path, f"layer_{index}")
        os.makedirs(layer_path, exist_ok=True)
        self.heuristics.save(layer_path)
        self.hippocampus.save(layer_path)
        self.proxy.save(layer_path)
        
        if self.next is not None:
            self.next.save(weight_path, index + 1)

    def load(self, weight_path, index=0):
        layer_path = os.path.join(weight_path, f"layer_{index}")
        self.heuristics.load(layer_path)
        self.hippocampus.load(layer_path)
        self.proxy.load(layer_path)

        if self.next is not None:
            self.next.load(weight_path, index + 1)

    async def incrementally_learn(self, path: List[Node]):
        if len(path) < 2:
            return

        await self.hippocampus.incrementally_learn(path)
        await self.proxy.incrementally_learn(path)

        entropies = await self.hippocampus.compute_entropy(path)
        pivots_indices = compute_pivot_indices(entropies)
        pivots = [path[i] for i in pivots_indices]

        await self.heuristics.incrementally_learn(path, pivots_indices)

        if self.next is not None and len(pivots) > 1:
            await self.next.incrementally_learn(pivots)

    async def to_next(self, c: Node, forward=True):
        # not just enhance, but select the closest with highest entropy

        entropy = (await self.hippocampus.compute_entropy([c]))[0]
        for i in range(1000):
            next_c = await self.hippocampus.sample(c, forward)
            next_entropy = (await self.hippocampus.compute_entropy([c]))[0]

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

        sup_info = [(await self.hippocampus.enhance(cortex_rep), cortex_prop), (await self.hippocampus.enhance(hippocampus_rep), hippocampus_prop)]

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
            goals = self.next.find_path(await self.to_next(c, True), await self.to_next(t, False))

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
                        result, data = await anext(goals)
                        if result:
                            yield False, data
                            break
                    g = await self.from_next(data)
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

                c, supplementary = await self.pincer_inference(c, g, pathway_bias)
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



async def build_cognitive_map(layers):
    hierarchy = []
    for i, layer_data in enumerate(layers):
        hierarchy.append(Layer(f"layer-{i}", layer_data["heuristics"], layer_data["hippocampus"], layer_data["proxy"]))
    for i in range(len(hierarchy) - 1):
        hierarchy[i].assign_next(hierarchy[i + 1])
    return hierarchy[0]


async def test():
    print("assert that probabilistic network works.")

    np.set_printoptions(precision=2)

    graph_shape = 16
    one_hot = generate_onehot_representation(np.arange(graph_shape), graph_shape)
    representations = [Node(one_hot[i, :]) for i in range(16)]

    config = {
        "layers": [
            {
                "heuristics": heuristic.Model(metric_network=resnet.Model(graph_shape), diminishing_factor=0.9, world_update_prior=0.1, reach=1, all_pairs=False),
                "hippocampus": hippocampus.Model(memory_size=128, chunk_size=graph_shape, diminishing_factor=0.9, embedding_dim=graph_shape),
                "proxy": proxy.Model(memory_size=128, chunk_size=graph_shape, candidate_count=graph_shape, embedding_dim=graph_shape)
            },
            {
                "heuristics": heuristic.Model(metric_network=resnet.Model(graph_shape), diminishing_factor=0.9, world_update_prior=0.1, reach=1, all_pairs=False),
                "hippocampus": hippocampus.Model(memory_size=128, chunk_size=graph_shape, diminishing_factor=0.9, embedding_dim=graph_shape),
                "proxy": proxy.Model(memory_size=128, chunk_size=graph_shape, candidate_count=graph_shape, embedding_dim=graph_shape)
            },
            {
                "heuristics": heuristic.Model(metric_network=resnet.Model(graph_shape), diminishing_factor=0.9, world_update_prior=0.1, reach=1, all_pairs=False),
                "hippocampus": hippocampus.Model(memory_size=128, chunk_size=graph_shape, diminishing_factor=0.9, embedding_dim=graph_shape),
                "proxy": proxy.Model(memory_size=128, chunk_size=graph_shape, candidate_count=graph_shape, embedding_dim=graph_shape)
            }
        ]
    }

    cognitive_map = await build_cognitive_map(**config)
    print(cognitive_map)


    dir_path = os.path.dirname(os.path.realpath(__file__))
    weight_path = os.path.join(dir_path, "..", "weights", "network.py.test")
    os.makedirs(weight_path, exist_ok=True)

    if os.path.exists(weight_path): 
        train = False
    else:
        train = True

    # read train from args
    train = args.retrain

    print("train:", train)

    if train:
        empty_directory(weight_path)

        graph = random_graph(graph_shape, 0.4)
        np.save(os.path.join(weight_path, "graph.npy"), graph)
        print(graph)

        explore_steps = 10000
        print("Training a cognitive map:")
        for i in range(explore_steps):
            path = random_walk(graph, random.randint(0, graph.shape[0] - 1), graph.shape[0] - 1)
            path = [representations[p] for p in path]
            await cognitive_map.incrementally_learn(path)
            if i % 100 == 0:
                print("Training progress: %.2f%%" % (i * 100 / explore_steps), end="\r", flush=True)
        print("\nFinish learning.")
        cognitive_map.save(weight_path)
    
    else:
        graph = np.load(os.path.join(weight_path, "graph.npy"))
        cognitive_map.load(weight_path)

    printer = Metric_Printer(Node_tensor_2D(graph_shape, 1, np.array([r.data for r in representations])))

    goals = range(graph_shape)
    max_steps = 40

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            path_generator = cognitive_map.find_path(representations[0], representations[t], hard_limit=max_steps)
            async for result, pi in path_generator:
                if not result:
                    await printer.print(pi)
                else:
                    await printer.print(np.argmax(pi), end=' ')
                    total_length = total_length + 1
        except RecursionError:
            await printer.print("fail to find path in time.", t, end=' ')
        finally:
            print()
    print("cognitive planner:", time.time() - stamp, " average length:", total_length / len(goals))
    print("======================================================")

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            path_generator = cognitive_map.find_path(representations[0], representations[t], hard_limit=max_steps, pathway_bias=-1)
            async for result, pi in path_generator:
                if not result:
                    await printer.print(pi)
                else:
                    await printer.print(np.argmax(pi), end=' ')
                    total_length = total_length + 1
        except RecursionError:
            await printer.print("fail to find path in time.", t, end=' ')
        finally:
            print()
    print("hippocampus planner:", time.time() - stamp, " average length:", total_length / len(goals))
    print("======================================================")

    total_length = 0
    stamp = time.time()
    for t in goals:
        try:
            path_generator = cognitive_map.find_path(representations[0], representations[t], hard_limit=max_steps, pathway_bias=1)
            async for result, pi in path_generator:
                if not result:
                    await printer.print(pi)
                else:
                    await printer.print(np.argmax(pi), end=' ')
                    total_length = total_length + 1
        except RecursionError:
            await printer.print("fail to find path in time.", t, end=' ')
        finally:
            await printer.print()
    print("cortex planner:", time.time() - stamp, " average length:", total_length / len(goals))

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




if __name__ == '__main__':
    from pathways import heuristic, hippocampus, proxy
    from metric import resnet
    from utilities import *
    import random
    asyncio.run(test())