import numpy as np
import os
import asyncio
from loguru import logger
import argparse
import random

from src.metric import Node, Node_tensor_2D, Metric_Printer, resnet
from src.network import Layer
from src.pathways import heuristic, hippocampus, proxy
from src.utilities import *

parser = argparse.ArgumentParser()
parser.add_argument("--retrain", action="store_true")
args = parser.parse_args()


async def build_cognitive_map(layers):
    hierarchy = []
    for i, layer_data in enumerate(layers):
        hierarchy.append(Layer(f"layer-{i}", layer_data["heuristics"], layer_data["hippocampus"], layer_data["proxy"]))
    for i in range(len(hierarchy) - 1):
        hierarchy[i].assign_next(hierarchy[i + 1])
    return hierarchy[0]


async def test():
    print("assert that probabilistic network works.")

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
    np.set_printoptions(precision=2)
    asyncio.run(test())