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


def clear_directory(output_dir):
    print("Clearing directory: {}".format(output_dir))
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            os.remove(os.path.join(root, file))


def compute_mse_loss(predictions, targets):
    return torch.mean(torch.square(predictions - targets))


def compute_contranstive_loss(predictions, targets):
    return compute_mse_loss(predictions, targets) - compute_mse_loss(predictions, torch.mean(predictions, dim=1, keepdim=True))


def compute_gausian_variational_loss(predictions, targets):
    X = targets - predictions
    mu = torch.mean(X, dim=1)
    mean_sqr = torch.mean(X * X, dim=1)
    var = mean_sqr - mu * mu
    return torch.mean(mean_sqr - torch.log(var))


log_2PI = math.log(2 * math.pi)


def compute_log_gausian_density_loss(x, means, variances):
    '''
        x of shape [dims, batch]
        means of shape [dims, batch]
        variances of shape [dims, batch]
    '''
    dims = x.shape[0]
    return torch.mean(torch.sum(torch.square(x - means) / (2 * variances), dim=0)
                      + 0.5 * torch.sum(torch.log(variances), dim=0)
                      + (0.5 * dims) * log_2PI
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


class Knapsack_model:
    pass


class Model:

    def __init__(self, dims, checkpoint_dir=None, save_every=None, num_epoch=100, lr=0.01, step_size=10, weight_decay=0.99):
        self.model = Knapsack_model(dims)
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

        # Setup the optimizers
        self.num_epoch = num_epoch
        self.save_format = "{:0" + str(len(str(num_epoch))) + "d}.ckpt"
        lr = lr

        parameters = list(self.model.parameters())
        self.opt = optim.Adam(parameters, lr=lr)
        self.step = 0
        self.step_size = step_size
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=weight_decay)

        self.eval_mode = True

    def consolidate(self, candidates, props, target):
        pass

    def incrementally_learn(self, path, pivots):
        self.model.train()

        path = torch.from_numpy(path) if type(path).__module__ == np.__name__ else path
        encoded_path = self.embedding_model.encode(path)
        V = encoded_path[:, :-1]
        H = encoded_path[:, 1:]
        neighbor_var = self.neighbor_variational_model(V)
        loss_values = compute_log_gausian_density_loss(H, V, neighbor_var)

        if pivots.size > 0:
            pivots = torch.from_numpy(pivots) if type(pivots).__module__ == np.__name__ else pivots
            P = encoded_path[:, pivots]
            heuristic_var = self.heuristic_variational_model(P)
            loss_values = loss_values + compute_log_gausian_density_loss_against_pivots(encoded_path, generate_masks(pivots, path.shape[1]), P, heuristic_var)
            loss_values = loss_values + 100 * compute_mse_loss(heuristic_var, 1)

        self.opt.zero_grad()
        loss_values.backward()
        self.opt.step()
        self.step += 1

        return loss_values.detach().cpu().numpy(), self.step

    def bootstrap(self, paths):
        clear_directory(self.checkpoint_dir)

        self.model.train()

        # Start training
        sum_loss = 0
        sum_count = 0

        writer = SummaryWriter(os.path.join(self.checkpoint_dir, "logs"))
        paths = torch.from_numpy(paths) if type(paths).__module__ == np.__name__ else paths

        while True:

            # Main training code
            loss, iterations = self.incrementally_learn(paths)
            sum_loss = sum_loss + loss
            sum_count = sum_count + 1

            writer.add_scalar('Loss/train', loss, iterations)

            if (iterations) % self.step_size == 0:
                self.scheduler.step()

            # Save network weights
            if (iterations) % self.save_every == 0:
                print("Iteration: %08d/%08d, loss: %.8f" % (iterations, self.num_epoch, sum_loss / sum_count))
                sum_loss = 0
                sum_count = 0
                self.save()

            if iterations >= self.num_epoch:
                break

    def load(self):
        if self.checkpoint_dir is not None:
            latest = sorted(os.listdir(os.path.join(self.checkpoint_dir, "weights")))[-1]
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, "weights", latest))
            self.model.load_state_dict(checkpoint['model'])
            self.opt.load_state_dict(checkpoint['opt'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.step = checkpoint['step']
        else:
            print("Cannot load weights, checkpoint_dir is None.")

    def save(self):
        if not os.path.exists(os.path.join(self.checkpoint_dir, "weights")):
            print("Creating directory: {}".format(os.path.join(self.checkpoint_dir, "weights")))
            os.makedirs(os.path.join(self.checkpoint_dir, "weights"))

        torch.save({
            'model': self.model.state_dict(),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step
        }, os.path.join(self.checkpoint_dir, "weights", self.save_format.format(self.step)))


if __name__ == '__main__':
    masks = generate_masks(torch.from_numpy(np.array([3, 5, 8])), 10)
    print(masks)

    print(compute_log_gausian_density_loss(torch.randn(4, 5), torch.randn(4, 5), torch.randn(4, 5)**2))
    print(compute_log_gausian_density_loss_against_pivots(torch.randn(4, 10), masks, torch.randn(4, 3), torch.randn(4, 3)**2))
