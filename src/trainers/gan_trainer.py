import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np


def clear_directory(output_dir):
    print("Clearing directory: {}".format(output_dir))
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            os.remove(os.path.join(root, file))


def compute_mse_loss(predictions, targets):
    return torch.mean(torch.square(predictions - targets))


def compute_gausian_density_loss(predictions, targets, log_density):
    return torch.mean(torch.square(predictions - targets) / 2) - torch.mean(log_density)


class Transposer(nn.Module):
    def __init__(self):
        super(Transposer, self).__init__()

    def __call__(self, x):
        if type(x) is tuple:
            return torch.transpose(x[0], 0, 1), x[1]
        else:
            return torch.transpose(x, 0, 1)


class Trainer:

    def __init__(self, embedding_model, checkpoint_dir=None, save_every=None, num_epoch=100, lr=0.01, step_size=10, weight_decay=0.99, on_cpu=True):
        self.embedding_model = embedding_model
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

        self.dims = self.embedding_model.dims()
        self.discriminator = nn.Sequential(
            Transposer(),
            nn.Linear(self.dims, self.dims),
            nn.ReLU(),
            nn.Linear(self.dims, self.dims),
            nn.ReLU(),
            nn.Linear(self.dims, 1),
            nn.Sigmoid(),
            Transposer()
        )

        self.criterion = nn.BCELoss()

        # Setup the optimizers
        self.num_epoch = num_epoch
        self.save_format = "{:0" + str(len(str(num_epoch))) + "d}.ckpt"
        lr = lr

        self.step = 0
        self.step_size = step_size
        self.gen_opt = optim.Adam(self.embedding_model.parameters(), lr=lr)
        self.gen_scheduler = optim.lr_scheduler.ExponentialLR(self.gen_opt, gamma=weight_decay)

        self.dis_opt = optim.Adam(self.discriminator.parameters(), lr=lr)
        self.dis_scheduler = optim.lr_scheduler.ExponentialLR(self.dis_opt, gamma=weight_decay)

        self.eval_mode = True

    def incrementally_learn(self, path):
        self.embedding_model.train()

        path = torch.from_numpy(path) if type(path).__module__ == np.__name__ else path
        encoded_path, log_density = self.embedding_model.encode_with_log_density(path)
        V = encoded_path[:, :-1]
        H = encoded_path[:, 1:]

        batch_size = path.shape[1] - 1

        displacement = H - V

        self.dis_opt.zero_grad()

        err_dis_real = self.criterion(
            torch.reshape(self.discriminator(displacement.detach()), [-1]),
            torch.zeros((batch_size,), dtype=torch.float))
        err_dis_real.backward()

        err_dis_fake = self.criterion(
            torch.reshape(self.discriminator(torch.randn(self.dims, batch_size)), [-1]),
            torch.ones((batch_size,), dtype=torch.float))
        err_dis_fake.backward()

        self.dis_opt.step()

        self.gen_opt.zero_grad()

        err_gen = self.criterion(
            torch.reshape(self.discriminator(displacement), [-1]),
            torch.ones((batch_size,), dtype=torch.float))
        err_gen.backward()

        self.gen_opt.step()

        self.step += 1

        return err_gen.detach().cpu().numpy(), self.step

    def bootstrap(self, path):
        pass

    def load(self):
        pass

    def save(self):
        pass


if __name__ == '__main__':
    pass
