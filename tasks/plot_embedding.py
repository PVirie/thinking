import numpy as np
import os
import asyncio
from loguru import logger
import argparse
# import plotly.express as px
# import pandas as pd
# from umap import UMAP

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

    if not os.path.exists(weight_path):
        print("No weights found. Run main.py to train.")
        return

    graph = np.load(os.path.join(weight_path, "graph.npy"))
    cognitive_map.load(weight_path)

    printer = Metric_Printer(Node_tensor_2D(graph_shape, 1, np.array([r.data for r in representations])))

    for i, layer in enumerate(config["layers"]):

        pairwise_dist = layer["heuristics"].metric_network.likelihood(representations, representations, cartesian=True)
        print(pairwise_dist)

        # umap_2d = UMAP(n_neighbors=10, n_components=2, init='random', random_state=0)
        # proj_2d = umap_2d.fit_transform(X_umap)

        # # on hover show prob
        # fig_2d = px.scatter(
        #     proj_2d, x='X', y='Y',
        #     hover_data=['Label']
        # )

        # plot_path = os.path.join(dir_path, "..", "plots", "network.py.test")
        
        # # make equal scale
        # fig_2d.update_xaxes(scaleanchor="y", scaleratio=1)
        # fig_2d.write_html(os.path.join(plot_path, f"layer-{i}.html"), auto_open=False)


if __name__ == '__main__':
    np.set_printoptions(precision=2)
    asyncio.run(test())