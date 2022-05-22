import variational_base
import torch
import torch.nn as nn
import numpy as np
import math


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


log_2PI = math.log(2 * math.pi)


class Model(variational_base.Model):

    @staticmethod
    def pincer_inference(neighbor_model, estimate_model, s, t):
        is_numpy = type(s).__module__ == np.__name__
        if is_numpy:
            s = torch.from_numpy(s)
            t = torch.from_numpy(t)

        var_n = neighbor_model(s)
        var_h = estimate_model(t)
        inferred_rep = (s * var_h + t * var_n) / (var_n + var_h)
        var = (var_n * var_h) / (var_n + var_h)
        dims = s.shape[0]
        inferred_prop = torch.exp(-log_2PI * dims * 0.5 + 0.5 * torch.sum(torch.log(var), dim=0))

        if is_numpy:
            return inferred_rep.detach().cpu().numpy(), inferred_prop.detach().cpu().numpy()

        return inferred_rep, inferred_prop

    def __init__(self, dims):
        self.dims = dims

        self.model = nn.Sequential(
            Transposer(),
            Residue_block(dims),
            Residue_block(dims),
            Residue_block(dims),
            Transposer()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # initialize any parameters
        # by default, the pre-built modules are already initialized.
        pass

    def dims(self):
        return self.dims

    def parameters(self):
        return self.model.parameters()

    def __call__(self, x):
        sd = self.model(x)
        return torch.clip(sd * sd, min=1e-6)

    def compute_entropy(self, x):
        is_numpy = type(x).__module__ == np.__name__
        x = torch.from_numpy(x) if is_numpy else x
        var = self.__call__(x)
        dims = x.shape[0]
        entropies = log_2PI * dims * 0.5 + 0.5 * torch.sum(torch.log(var), dim=0) + 0.5 * dims
        return entropies.detach().cpu().numpy() if is_numpy else entropies

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def state_dict(self):
        return self.model.state_dict()


if __name__ == '__main__':

    model = Model(8)

    means = np.random.normal(0, 1.0, [8, 16]).astype(np.float32)
    means = torch.from_numpy(means)
    print(model(means).shape)
