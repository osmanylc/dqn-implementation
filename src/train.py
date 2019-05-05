import torch
from torch import nn


def make_y(transitions, agt):
    y = []

    for phi_t, a_t, r_t, phi_t1, done in transitions:
        if done:
            y.append(r_t)
        else:
            x = phi_t.unsqueeze(0)
            y.append(r_t + .99 * agt.get_best_values(x).item())

    return torch.tensor(y, dtype=torch.float, device=agt.device)


def get_max_vals(transitions, agt):
    phis = []

    for phi_t, a_t, r_t, phi_t1, done in transitions:
        phis.append(phi_t)

    x = torch.stack(phis)
    return agt.get_best_values(x)


def mini_batch_loss(mini_batch, agt):
    y = make_y(mini_batch, agt)
    qmax = get_max_vals(mini_batch, agt)

    loss = nn.MSELoss(reduction='mean')
    return loss(y, qmax)
