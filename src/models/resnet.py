import torch
import torch.nn as nn
import os
import embedding_base
import numpy as np


class Transposer(nn.Module):
    def __init__(self):
        super(Transposer, self).__init__()

    def __call__(self, x):
        if type(x) is tuple:
            return torch.transpose(x[0], 0, 1), x[1]
        else:
            return torch.transpose(x, 0, 1)


class Residue_block(nn.Module):
    def __init__(self, dims):
        super(Residue_block, self).__init__()
        self.model1 = nn.Linear(dims, dims)
        self.model2 = nn.Linear(dims, dims)

    def __call__(self, x):
        y = self.model2(nn.functional.relu6(self.model1(x))) + x
        return nn.functional.relu6(y)

class Model(embedding_base.Model):

    def __init__(self, dims):
        self.input_dims = dims

        self.forward = nn.Sequential(
            Transposer(),
            Residue_block(dims),
            Residue_block(dims),
            Residue_block(dims),
            Transposer()
        )
        self.backward = nn.Sequential(
            Transposer(),
            Residue_block(dims),
            Residue_block(dims),
            Residue_block(dims),
            Transposer()
        )

        self.reset_parameters()

    def dims(self):
        return self.input_dims

    def reset_parameters(self) -> None:
        # initialize any parameters
        # by default, the pre-built modules are already initialized.
        pass

    def parameters(self):
        return list(self.forward.parameters()) + list(self.backward.parameters())

    def encode(self, c):
        is_numpy = type(c).__module__ == np.__name__
        result = self.forward(torch.from_numpy(c) if is_numpy else c)
        return result.detach().cpu().numpy() if is_numpy else result

    def decode(self, h):
        is_numpy = type(h).__module__ == np.__name__
        result = self.backward(torch.from_numpy(h) if is_numpy else h)
        return result.detach().cpu().numpy() if is_numpy else result

    def train(self):
        self.forward.train()
        self.backward.train()

    def eval(self):
        self.forward.eval()
        self.backward.eval()

    def load_state_dict(self, state_dict):
        self.forward.load_state_dict(state_dict["forward"])
        self.backward.load_state_dict(state_dict["backward"])

    def state_dict(self):
        state_dict = {
            "forward": self.forward.state_dict(),
            "backward": self.backward.state_dict()
        }
        return state_dict


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    model = Model(8)
    print(model.parameters())

    path = np.random.normal(0, 1.0, [8, 16]).astype(np.float32)
    path = torch.from_numpy(path)
    print(model.encode(path).shape)
