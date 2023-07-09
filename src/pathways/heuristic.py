import sys
import os
import math
import os
import numpy as np
import asyncio
from typing import List
from loguru import logger
from base import Pathway, Node


log_2PI = math.log(2 * math.pi)


def generate_masks(pivots, length, diminishing_factor=0.9, pre_steps=1):
    pos = np.expand_dims(np.arange(0, length, dtype=np.int32), axis=1)
    pre_pivots = pivots
    for i in range(pre_steps):
        pre_pivots = np.roll(pre_pivots, 1, 0)
        pre_pivots[0] = -1
    masks = np.logical_and(pos > np.expand_dims(pre_pivots, axis=0), pos <= np.expand_dims(pivots, axis=0)).astype(np.float32)

    order = np.reshape(np.arange(0, -length, -1), [-1, 1]) + np.expand_dims(pivots, axis=0)
    diminishing = np.power(diminishing_factor, order)
    return masks, diminishing


class Model(Pathway):

    def __init__(self, metric_network, diminishing_factor, world_update_prior, reach=1, all_pairs=False):
        self.metric_network = metric_network
        self.diminishing_factor = diminishing_factor
        self.new_target_prior = world_update_prior
        self.reach = reach  # how many pivots to reach. the larger the coverage, the longer the training.
        self.no_pivot = all_pairs  # extreme training condition all node to all nodes


    async def consolidate(self, start: Node, candidates: List[Node], props: List[float], target: Node):
        # candidates has shape [num_memory]
        # props has shape [num_memory]

        num_mem = len(candidates)

        heuristic_scores = self.metric_network.distance(candidates, target)

        scores = props * heuristic_scores
        max_candidate = np.argmax(scores)

        heuristic_rep = candidates[max_candidate]
        # should we recompute score by feeding it back to the nextwork instead of haphazard method?
        heuristic_prop = scores[max_candidate]

        base_scores = self.metric_network.distance(start, target)
        if heuristic_prop < base_scores:
            heuristic_prop = 0

        # print("5 best", torch.argmax(heuristic_rep, dim=0))
        return heuristic_rep, heuristic_prop

    async def incrementally_learn(self, path: List[Node], pivots):

        reach = self.reach
        path_length = len(path)

        # learn self
        # self_divergences = self.metric_network.distance(path[pivots], path[pivots])
        self_target = 1.0
        self_weight = 0.1
        self.metric_network.learn([path[p] for p in pivots], [path[p] for p in pivots], self_target, self_weight)

        if self.no_pivot:
            pivots = range(path_length)
            reach = path_length

        if len(pivots) > 0:
            masks, new_targets = generate_masks(pivots, path_length, self.diminishing_factor, reach)
            s = path
            t = [path[p] for p in pivots]
            divergences = self.metric_network.distance(s, t) 
            # divergence is a matrix of shape [len(t), len(s)]
            targets = np.where(new_targets < divergences, new_targets, (1 - self.new_target_prior) * divergences + self.new_target_prior * new_targets)
            
            # loss_values += torch.mean(masks * torch.square(divergences - targets))
            self.metric_network.learn(s, t, targets, masks)



if __name__ == '__main__':
    masks = generate_masks([3, 5, 8], 10, pre_steps=1)
    print(masks)

    # To do: write test
