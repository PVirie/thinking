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

        self.model_from = nn.Sequential(
            Transposer(),
            Residue_block(self.input_dims),
            Residue_block(self.input_dims),
            Residue_block(self.input_dims),
            Transposer()
        )

        self.model_to = nn.Sequential(
            Transposer(),
            Residue_block(self.input_dims),
            Residue_block(self.input_dims),
            Residue_block(self.input_dims),
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
        return list(self.model_from.parameters()) + list(self.model_to.parameters())

    def compute_divergence(self, a, b, return_numpy=True):
        a = torch.from_numpy(a) if type(a).__module__ == np.__name__ else a
        b = torch.from_numpy(b) if type(b).__module__ == np.__name__ else b

        result = torch.sigmoid(torch.sum(self.model_from(a) * self.model_to(b), dim=0, keepdims=True))
        return result.detach().cpu().numpy() if return_numpy else result

    def train(self):
        self.model_from.train()
        self.model_to.train()

    def eval(self):
        self.model_from.eval()
        self.model_to.eval()

    def load_state_dict(self, state_dict):
        self.model_from.load_state_dict(state_dict["from"])
        self.model_to.load_state_dict(state_dict["to"])

    def state_dict(self):
        state_dict = {
            "from": self.model_from.state_dict(),
            "to": self.model_to.state_dict()
        }
        return state_dict


if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))

    model = Model(8)

    path = np.random.normal(0, 1.0, [8, 16]).astype(np.float32)
    path = torch.from_numpy(path)
    print(model.compute_divergence(path, path).shape)
