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


class Trainer:

    def __init__(self, embedding_model, neighbor_model, checkpoint_dir=None, save_every=None, num_epoch=100, lr=0.01, step_size=10, weight_decay=0.99, on_cpu=True):
        self.embedding_model = embedding_model
        self.neighbor_model = neighbor_model
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

        # Setup the optimizers
        self.num_epoch = num_epoch
        self.save_format = "{:0" + str(len(str(num_epoch))) + "d}.ckpt"
        lr = lr

        self.opt = optim.Adam(list(self.embedding_model.parameters()) + list(self.neighbor_model.parameters()), lr=lr)
        self.step = 0
        self.step_size = step_size
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=weight_decay)

        self.eval_mode = True

    def incrementally_learn(self, path):
        self.embedding_model.train()
        self.neighbor_model.train()

        path = torch.from_numpy(path) if type(path).__module__ == np.__name__ else path
        encoded_path = self.embedding_model.encode(path)
        V = encoded_path[:, :-1]
        H = encoded_path[:, 1:]

        loss_values = compute_mse_loss(self.neighbor_model.forward(V), H)

        self.opt.zero_grad()
        loss_values.backward()
        self.opt.step()
        self.step += 1

        return loss_values, self.step

    def bootstrap(self, path):
        clear_directory(self.checkpoint_dir)

        self.embedding_model.train()
        self.neighbor_model.train()

        # Start training
        sum_loss = 0
        sum_count = 0

        writer = SummaryWriter(os.path.join(self.checkpoint_dir, "logs"))
        path = torch.from_numpy(path) if type(path).__module__ == np.__name__ else path

        while True:

            # Main training code
            loss, iterations = self.incrementally_learn(path)
            sum_loss = sum_loss + loss.detach().cpu().numpy()
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
            self.embedding_model.load_state_dict(checkpoint['embedding_model'])
            self.neighbor_model.load_state_dict(checkpoint['neighbor_model'])
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
            'embedding_model': self.embedding_model.state_dict(),
            'neighbor_model': self.neighbor_model.state_dict(),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step
        }, os.path.join(self.checkpoint_dir, "weights", self.save_format.format(self.step)))


if __name__ == '__main__':
    pass
