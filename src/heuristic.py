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


def compute_log_gausian_density_loss_against_pivots(x, masks, P, variances):
    '''
        x of shape [dims, length]
        masks of shape [length, batch]
        P of shape [dims, batch]
        variances of shape [dims, batch]
    '''
    x = torch.unsqueeze(x, dim=2)
    P = torch.unsqueeze(P, dim=1)

    dims = x.shape[0]
    return torch.mean(
        torch.sum(
            masks * torch.sum(torch.square(x - P) / (2 * torch.unsqueeze(variances, dim=1)), dim=0),
            dim=0) / torch.sum(masks, dim=0)
        + 0.5 * torch.sum(torch.log(variances), dim=0)
        + (0.5 * dims) * log_2PI
    )


def generate_masks(pivots, length):
    pos = torch.unsqueeze(torch.arange(0, length, dtype=torch.int32), dim=1)
    pre_pivots = torch.roll(pivots, 1, 0)
    pre_pivots[0] = -1
    return torch.logical_and(pos > torch.unsqueeze(pre_pivots, dim=0), pos <= torch.unsqueeze(pivots, dim=0)).float()


class Model:

    def __init__(self, dims, lr=0.01, step_size=10, weight_decay=0.99):
        self.model = embeddings.spline_flow.Model(dims)
        self.metric = metrics.euclidean.Model(dims)

        # Setup the optimizers
        parameters = list(self.model.parameters())
        self.opt = optim.Adam(parameters, lr=lr)
        self.step = 0
        self.step_size = step_size
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=weight_decay)

        self.eval_mode = True

    def consolidate(self, candidates, props, target, return_numpy=True):
        # candidates has shape [dim, batch]
        # props has shape [batch]
        # target has shape [dim, 1]

        candidates = torch.from_numpy(candidates) if type(candidates).__module__ == np.__name__ else candidates
        target = torch.from_numpy(target) if type(target).__module__ == np.__name__ else target
        props = torch.from_numpy(props) if type(props).__module__ == np.__name__ else props

        encoded_candidates = self.model.encode(candidates)
        encoded_target = self.model.encode(target)
        var = 1.0

        heuristic_scores = torch.exp(-0.5 * self.metric.sqr_dist(encoded_candidates, encoded_target) / var)

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
        encoded_path = self.model.encode(path)

        if pivots.size > 0:
            pivots = torch.from_numpy(pivots) if type(pivots).__module__ == np.__name__ else pivots
            P = encoded_path[:, pivots]
            loss_values = compute_loss_against_pivots(encoded_path, generate_masks(pivots, path.shape[1]), P, self.metric)

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

    print(compute_log_gausian_density_loss_against_pivots(torch.randn(4, 10), masks, torch.randn(4, 3), torch.randn(4, 3)**2))

    model = Model(2)

    candidates = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    props = np.array([1.0, 1.0], dtype=np.float32)
    targets = np.array([[1.0], [1.0]], dtype=np.float32)

    results = model.consolidate(candidates, props, targets)
    print(results)
