import torch
import torch.nn as nn
import modules

import os
import embedding_base
import numpy as np


class Transposer(nn.Module):
    def __init__(self):
        super(Transposer, self).__init__()

    def __call__(self, x):
        return torch.transpose(x, 0, 1)


class Forward_interface(nn.Module):
    def __init__(self, model):
        super(Forward_interface, self).__init__()
        self.model = model

    def __call__(self, x):
        return self.model.forward(x)[0]


class Backward_interface(nn.Module):
    def __init__(self, model):
        super(Backward_interface, self).__init__()
        self.model = model

    def __call__(self, x):
        return self.model.inverse(x)[0]


class Model(embedding_base.Model):

    def __init__(self, dims):
        self.input_dims = dims

        self.model = modules.NSF_CL(self.input_dims, K=5, B=3, hidden_dim=8)

        self.forward = nn.Sequential(
            Transposer(),
            Forward_interface(self.model),
            Transposer()
        )
        self.backward = nn.Sequential(
            Transposer(),
            Backward_interface(self.model),
            Transposer()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # initialize any parameters
        # by default, the pre-built modules are already initialized.
        pass

    def parameters(self):
        return self.model.parameters()

    def encode(self, c):
        is_numpy = type(c).__module__ == np.__name__
        result = self.forward(torch.from_numpy(c) if is_numpy else c)
        return result.detach().cpu().numpy() if is_numpy else result

    def decode(self, h):
        is_numpy = type(h).__module__ == np.__name__
        result = self.backward(torch.from_numpy(h) if is_numpy else h)
        return result.detach().cpu().numpy() if is_numpy else result

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    model = Model(8)

    path = np.random.normal(0, 1.0, [8, 16]).astype(np.float32)
    path = torch.from_numpy(path)
    print(model.encode(path).shape)
