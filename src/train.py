import torch
from torch import nn


def make_y(transitions, agt):
    y = []

    for phi_t, a_t, r_t, phi_t1, done in transitions:
        if done:
            y.append(r_t)
        else:
            x = phi_t1.unsqueeze(0).to(agt.device)
            y_i = r_t + .99 * agt.target(x).max().item()
            y.append(y_i)

    return torch.tensor(y, dtype=torch.float, device=agt.device)


def get_act_vals(transitions, agt):
    phis = []
    acts = []

    for phi_t, a_t, r_t, phi_t1, done in transitions:
        phis.append(phi_t)
        acts.append(a_t)

    x = torch.stack(phis).to(agt.device)
    acts = torch.tensor(acts, dtype=torch.long).unsqueeze(1).to(agt.device)
    return agt.qnet(x).gather(1, acts)


def mini_batch_loss(mini_batch, agt):
    y = make_y(mini_batch, agt)
    qmax = get_act_vals(mini_batch, agt)

    loss = nn.MSELoss()
    return loss(y, qmax)
