import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import os
import numpy as np
import embeddings
import metrics


log_2PI = math.log(2 * math.pi)


def compute_loss_against_pivots(x, masks, P, metric):
    '''
        x of shape [dims, length]
        masks of shape [length, batch]
        P of shape [dims, batch]
    '''
    x = torch.unsqueeze(x, dim=2)
    P = torch.unsqueeze(P, dim=1)

    return torch.mean(
        torch.sum(
            masks * metric.sqr_dist(x, P),
            dim=0) / torch.sum(masks, dim=0)
    )


def generate_masks(pivots, length, diminishing_factor=0.9):
    pos = torch.unsqueeze(torch.arange(0, length, dtype=torch.int32), dim=1)
    pre_pivots = torch.roll(pivots, 1, 0)
    pre_pivots[0] = -1
    masks = torch.logical_and(pos > torch.unsqueeze(pre_pivots, dim=0), pos <= torch.unsqueeze(pivots, dim=0)).float()

    order = torch.reshape(torch.arange(0, -length, -1), [-1, 1]) + torch.unsqueeze(pivots, dim=0)
    diminishing = torch.pow(diminishing_factor, order)
    return masks, diminishing


class Model:

    def __init__(self, dims, lr=0.01, step_size=10, weight_decay=0.99):
        self.model = embeddings.divergence.Model(dims)
        self.metric = metrics.euclidean.Model(dims)

        # Setup the optimizers
        parameters = list(self.model.parameters())
        self.opt = optim.Adam(parameters, lr=lr)
        self.step = 0
        self.step_size = step_size
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=weight_decay)

        self.eval_mode = True

    def dist(self, s, t, return_numpy=True):
        s = torch.from_numpy(s) if type(s).__module__ == np.__name__ else s
        t = torch.from_numpy(t) if type(t).__module__ == np.__name__ else t

        results = torch.sum(self.model.compute_divergence(s, torch.tile(t, (1, s.shape[1])), return_numpy=False), dim=0)

        if return_numpy:
            return results.detach().cpu().numpy()
        else:
            return results

    def consolidate(self, candidates, props, target, return_numpy=True):
        # candidates has shape [dim, batch]
        # props has shape [batch]
        # target has shape [dim, 1]

        candidates = torch.from_numpy(candidates) if type(candidates).__module__ == np.__name__ else candidates
        target = torch.from_numpy(target) if type(target).__module__ == np.__name__ else target
        props = torch.from_numpy(props) if type(props).__module__ == np.__name__ else props

        target = torch.tile(target, (1, candidates.shape[1]))
        heuristic_scores = self.model.compute_divergence(candidates, target, return_numpy=False)

        nominators = props * heuristic_scores
        weights = nominators / torch.sum(nominators)

        result = torch.sum(candidates * weights, dim=1, keepdims=True)

        if return_numpy:
            return result.detach().cpu().numpy(), np.array([1.0], dtype=np.float32)
        else:
            return result, np.array([1.0], dtype=np.float32)

    def incrementally_learn(self, path, pivots):
        self.model.train()

        path = torch.from_numpy(path) if type(path).__module__ == np.__name__ else path

        if pivots.size > 0:
            pivots = torch.from_numpy(pivots) if type(pivots).__module__ == np.__name__ else pivots

            masks, targets = generate_masks(pivots, path.shape[1])
            s = torch.reshape(torch.tile(torch.unsqueeze(path, dim=2), (1, 1, pivots.shape[0])), [path.shape[0], -1])
            t = torch.reshape(torch.tile(torch.unsqueeze(path[:, pivots], dim=1), (1, path.shape[1], 1)), [path.shape[0], -1])
            divergences = torch.reshape(self.model.compute_divergence(s, t, return_numpy=False), [path.shape[1], pivots.shape[0]])

            loss_values = torch.mean(torch.square(divergences - targets))

        self.opt.zero_grad()
        loss_values.backward()
        self.opt.step()
        self.step += 1

        return loss_values.detach().cpu().numpy(), self.step

    def load(self, checkpoint_dir):
        if checkpoint_dir is None:
            print("Cannot load weights, checkpoint_dir is None.")
            return

        weight_dir = os.path.join(checkpoint_dir, "weights")

        if not os.path.exists(weight_dir):
            print("Cannot load weights, weights do not exists.")
            return

        latest = sorted(os.listdir(weight_dir))[-1]
        checkpoint = torch.load(os.path.join(weight_dir, latest))
        self.model.load_state_dict(checkpoint['model'])
        self.opt.load_state_dict(checkpoint['opt'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.step = checkpoint['step']

    def save(self, checkpoint_dir):

        if checkpoint_dir is None:
            print("Cannot save weights, checkpoint_dir is None.")
            return

        weight_dir = os.path.join(checkpoint_dir, "weights")
        if not os.path.exists(weight_dir):
            print("Creating directory: {}".format(weight_dir))
            os.makedirs(weight_dir)

        save_format = "{:6d}.ckpt"
        torch.save({
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step
        }, os.path.join(weight_dir, save_format.format(self.step)))


if __name__ == '__main__':
    masks = generate_masks(torch.from_numpy(np.array([3, 5, 8])), 10)
    print(masks)

    model = Model(2)

    candidates = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    props = np.array([1.0, 1.0], dtype=np.float32)
    targets = np.array([[1.0], [1.0]], dtype=np.float32)

    results = model.consolidate(candidates, props, targets)
    print(results)

    #############################################################

    from utilities import *

    model = Model(8, lr=0.01, step_size=100, weight_decay=0.99)

    graph = random_graph(8, 0.5)
    print(graph)
    all_reps = generate_onehot_representation(np.arange(graph.shape[0]), graph.shape[0])
    explore_steps = 1000

    stat_graph = np.zeros([graph.shape[0]], dtype=np.float32)
    position = (np.power(0.9, np.arange(graph.shape[0], 0, -1) - 1))

    for i in range(explore_steps):
        path = random_walk(graph, 0, graph.shape[0] - 1)
        path.reverse()

        stat_graph[path] = stat_graph[path] - position[:len(path)]

        encoded = all_reps[:, path]
        loss, _ = model.incrementally_learn(encoded, np.array([encoded.shape[1] - 1], dtype=np.int64))
        if i % (explore_steps // 100) == 0:
            print("Training progress: %.2f%% %.8f" % (i * 100 / explore_steps, loss), end="\r", flush=True)

    print(stat_graph, np.argsort(stat_graph))

    dists = model.dist(all_reps, all_reps[:, 0:1])
    print(dists, np.argsort(-dists))