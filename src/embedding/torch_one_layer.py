import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import embedding_base
import numpy as np


class Linear(nn.Module):
    def __init__(self, input_dims, output_dims, device):
        super(Linear, self).__init__()
        self.W = nn.Linear(input_dims, output_dims, device=device)

    def __call__(self, x):
        return torch.transpose(self.W(torch.transpose(x, 0, 1)), 1, 0)


def compute_mse_loss(predictions, targets):
    return torch.mean(torch.square(predictions - targets))


def clear_directory(output_dir):
    print("Clearing directory: {}".format(output_dir))
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            os.remove(os.path.join(root, file))


class Embedding(embedding_base.Embedding):

    def __init__(self, input_dims, output_dims, checkpoint_dir=None, save_every=None, num_epoch=100, lr=0.01, step_size=10, weight_decay=0.99, on_cpu=True):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every

        self.device = torch.device("cuda:0") if not on_cpu else torch.device("cpu:0")

        self.forward = Linear(self.input_dims, self.output_dims, self.device)
        self.backward = Linear(self.output_dims, self.input_dims, self.device)

        # Setup the optimizers
        self.num_epoch = num_epoch
        self.save_format = "{:0" + str(len(str(num_epoch))) + "d}.ckpt"
        lr = lr

        self.opt = optim.Adam(list(self.forward.parameters()) + list(self.backward.parameters()), lr=lr)
        self.step = 0
        self.step_size = step_size
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.opt, gamma=weight_decay)

        self.eval_mode = True

    def set_eval_mode(self, flag):
        if flag == self.eval_mode:
            return
        if flag:
            self.forward.eval()
            self.backward.eval()
        else:
            self.forward.train()
            self.backward.train()

    def update(self, V, H):

        V_ = self.forward(V)
        proximity_loss = compute_mse_loss(V_, self.forward(H))
        reconstruction_loss = compute_mse_loss(self.backward(V_), V)
        loss_values = proximity_loss + reconstruction_loss

        self.opt.zero_grad()
        loss_values.backward()
        self.opt.step()
        self.step += 1

        return loss_values, self.step

    def encode(self, c):
        self.set_eval_mode(True)
        to_numpy = type(c).__module__ == np.__name__
        c = torch.from_numpy(c).to(self.device)
        result = self.forward(c)
        return result.detach().cpu().numpy() if to_numpy else result

    def decode(self, h):
        self.set_eval_mode(True)
        to_numpy = type(h).__module__ == np.__name__
        h = torch.from_numpy(h).to(self.device)
        result = self.backward(h)
        return result.detach().cpu().numpy() if to_numpy else result

    def load(self):
        if self.checkpoint_dir is not None:
            latest = sorted(os.listdir(os.path.join(self.checkpoint_dir, "weights")))[-1]
            checkpoint = torch.load(os.path.join(self.checkpoint_dir, "weights", latest))
            self.forward.load_state_dict(checkpoint['forward'])
            self.backward.load_state_dict(checkpoint['backward'])
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
            'forward': self.forward.state_dict(),
            'backward': self.backward.state_dict(),
            'opt': self.opt.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'step': self.step
        }, os.path.join(self.checkpoint_dir, "weights", self.save_format.format(self.step)))

    def incrementally_learn(self, path):
        self.set_eval_mode(False)

        path = torch.from_numpy(path).to(self.device)
        loss, iterations = self.update(path[:, :-1], path[:, 1:])

    def bootstrap(self, path):
        clear_directory(self.checkpoint_dir)

        self.set_eval_mode(False)

        # Start training
        sum_loss = 0
        sum_count = 0

        writer = SummaryWriter(os.path.join(self.checkpoint_dir, "logs"))

        path = torch.from_numpy(path).to(self.device)
        V = torch.reshape(path[:, :-1, :], [self.input_dims, -1])
        H = torch.reshape(path[:, 1:, :], [self.input_dims, -1])
        while True:

            # Main training code
            loss, iterations = self.update(V, H)
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


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    model = Embedding(**{
        'input_dims': 8, 'output_dims': 4,
        'checkpoint_dir': os.path.join(dir_path, "..", "..", "artifacts", "torch_one_layer"),
        'save_every': 10, 'num_epoch': 100,
        'lr': 0.01, 'step_size': 10, 'weight_decay': 0.95,
        'on_cpu': False
    })

    path = np.random.normal(0, 1.0, [8, 16, 2]).astype(np.float32)
    model.bootstrap(path)
