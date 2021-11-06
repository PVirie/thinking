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


def compute_n_sphere_surface_with_gaussian_energy_probability(neighbor_radius, var_h, distance, x_t_2, dims):
    return torch.exp((distance**2 + neighbor_radius**2 - x_t_2) / (2 * var_h)) / (((2 * math.pi) ** (dims - 1)) * torch.i0(neighbor_radius * distance / var_h))


class Model(variational_base.Model):

    @staticmethod
    def pincer_inference(neighbor_model, estimate_model, s, t):
        is_numpy = type(s).__module__ == np.__name__
        if is_numpy:
            s = torch.from_numpy(s)
            t = torch.from_numpy(t)

        neighbor_radius = neighbor_model(s)
        var_h = estimate_model(t)
        dims = s.shape[0]
        t_s = torch.sqrt(torch.sum((t - s)**2, dim=0, keepdim=True))
        inferred_rep = s + (t - s) * neighbor_radius / t_s
        x_t_2 = torch.sum((t - inferred_rep)**2, dim=0, keepdim=True)
        inferred_prop = torch.squeeze(compute_n_sphere_surface_with_gaussian_energy_probability(neighbor_radius, var_h, t_s, x_t_2, dims), dim=0)

        if is_numpy:
            return inferred_rep.detach().cpu().numpy(), inferred_prop.detach().cpu().numpy()

        return inferred_rep, inferred_prop

    def __init__(self, dims):
        self.num_dims = dims

        self.model = nn.Sequential(
            Transposer(),
            Residue_block(dims),
            Residue_block(dims),
            Residue_block(dims),
            nn.Linear(dims, 1),
            Transposer()
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # initialize any parameters
        # by default, the pre-built modules are already initialized.
        pass

    def dims(self):
        return self.num_dims

    def parameters(self):
        return self.model.parameters()

    def __call__(self, x):
        hr = self.model(x)
        return torch.clip(hr * hr, min=1e-6) + 1

    def compute_entropy(self, x):
        is_numpy = type(x).__module__ == np.__name__
        x = torch.from_numpy(x) if is_numpy else x
        radius = self.__call__(x)
        return radius.detach().cpu().numpy() if is_numpy else radius

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

    means = np.random.normal(0, 1.0, [model.dims(), 16]).astype(np.float32)
    means = torch.from_numpy(means)
    print(model(means).shape)
    print(model.pincer_inference(model, model, means, means + 2))
