import os
import math
import jax.numpy as jnp
from jax import jit
from typing import List
from loguru import logger
from base import Pathway, Node


log_2PI = math.log(2 * math.pi)


def generate_masks(pivots, length, diminishing_factor=0.9, pre_steps=1):
    pivots = jnp.array(pivots, dtype=jnp.int32)
    pos = jnp.expand_dims(jnp.arange(0, length, dtype=jnp.int32), axis=1)
    
    # pre_pivots = pivots
    # for i in range(pre_steps):
    #     pre_pivots = jnp.roll(pre_pivots, pre_steps, 0)
    #     pre_pivots[: pre_steps] = -1
    # use concatenate instead
    pre_pivots = jnp.concatenate([jnp.full([pre_steps], -1, dtype=jnp.int32), pivots[:-pre_steps]], axis=0)

    masks = jnp.logical_and(pos > jnp.expand_dims(pre_pivots, axis=0), pos <= jnp.expand_dims(pivots, axis=0)).astype(jnp.float32)

    order = jnp.reshape(jnp.arange(0, -length, -1), [-1, 1]) + jnp.expand_dims(pivots, axis=0)
    diminishing = jnp.power(diminishing_factor, order)
    return jnp.transpose(masks), jnp.transpose(diminishing)


def generate_q_learning(current_values, length, diminishing_factor=0.9):
    # current_values is a matrix of shape [batch, length]
    current_values = jnp.array(current_values)
    xv, yv = jnp.meshgrid(jnp.linspace(0, - length + 1, length), jnp.linspace(0, length - 1, length))
    mesh = xv + yv
    diminishing = jnp.power(diminishing_factor, mesh)
    diminishing = jnp.where(mesh < 0, 0, diminishing)

    # https://stackoverflow.com/questions/41164305/numpy-dot-product-with-max-instead-of-sum
    M3 = jnp.max(current_values[:, :, None] * diminishing[None, :, :], axis = 1)
    return M3


class Model(Pathway):

    def __init__(self, metric_network, diminishing_factor, world_update_prior, reach=1, all_pairs=False, name=""):
        self.metric_network = metric_network
        self.diminishing_factor = diminishing_factor
        self.world_update_prior = world_update_prior
        self.reach = reach  # how many pivots to reach. the larger the coverage, the longer the training.
        self.no_pivot = all_pairs  # extreme training condition all node to all nodes

        self.name = name
        self.learn_steps = 0


    def save(self, path):
        weight_path = os.path.join(path, "heuristic")
        os.makedirs(weight_path, exist_ok=True)
        self.metric_network.save(weight_path)


    def load(self, path):
        weight_path = os.path.join(path, "heuristic")
        self.metric_network.load(weight_path)


    async def consolidate(self, start: Node, candidates: List[Node], props: List[float], target: Node):
        # candidates has shape [num_memory]
        # props has shape [num_memory]

        num_mem = len(candidates)

        heuristic_scores = self.metric_network.likelihood(candidates, target)

        scores = jnp.array(props, jnp.float32) * heuristic_scores
        max_candidate = jnp.argmax(scores)

        heuristic_rep = candidates[max_candidate]
        # should we recompute score by feeding it back to the nextwork instead of haphazard method?
        heuristic_prop = scores[max_candidate]

        base_scores = self.metric_network.likelihood(start, target)
        # if heuristic_prop < base_scores:
        #     heuristic_prop = 0

        # print("5 best", torch.argmax(heuristic_rep, dim=0))
        return heuristic_rep, heuristic_prop


    async def incrementally_learn(self, path: List[Node], pivots):

        reach = self.reach
        path_length = len(path)

        # learn self
        # self_divergences = self.metric_network.likelihood(path[pivots], path[pivots])
        self_label = 1.0
        self_weight = 0.1
        self.metric_network.learn([path[p] for p in pivots], [path[p] for p in pivots], self_label, self_weight)

        if self.no_pivot:
            pivots = range(path_length)
            reach = path_length

        if len(pivots) > 0:
            masks, new_scores = generate_masks(pivots, path_length, self.diminishing_factor, reach)
            s = path
            t = [path[p] for p in pivots]
            # current_scores = self.metric_network.likelihood(s, t, cartesian=True)
            # current_scores is a matrix of shape [len(t), len(s)]

            labels = new_scores
            # labels = jnp.where(new_scores > current_scores, new_scores, (1 - self.world_update_prior) * current_scores + self.world_update_prior * new_scores)
            # labels = generate_q_learning(current_scores, path_length, self.diminishing_factor)

            loss = self.metric_network.learn(s, t, labels, masks, cartesian=True)

            self.learn_steps += 1
            if self.learn_steps % 100 == 0:
                logger.info(f"heuristic {self.name} loss: {loss}")


    async def incrementally_learn_2(self, path: List[Node], pivots, hippocampus_distances):

        reach = self.reach
        path_length = len(path)

        # learn self
        self_label = 1.0
        self_weight = 0.1
        self.metric_network.learn([path[p] for p in pivots], [path[p] for p in pivots], self_label, self_weight)

        if len(pivots) > 0:
            masks, new_scores = generate_masks(pivots, path_length, self.diminishing_factor, reach)
            s = path
            t = [path[p] for p in pivots]

            loss = self.metric_network.learn(s, t, hippocampus_distances, masks, cartesian=True)

            self.learn_steps += 1
            if self.learn_steps % 100 == 0:
                logger.info(f"heuristic {self.name} loss: {loss}")


if __name__ == '__main__':
    masks, labels = generate_masks([3, 5, 8], 10, pre_steps=1)
    print(masks)
    print(labels)

    kernel = generate_q_learning([[0, 1, 2], [0, 2, 1]], 3)
    print(kernel)