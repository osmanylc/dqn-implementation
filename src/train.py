import sys
sys.path.append('.')

import time

import gym
import torch
from torch import nn
from torch import optim

# import agent, utils


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


# def train():
#     replay_mem_size = int(5e5)
#     mini_batch_size = 64
#     num_episodes = int(20)
#
#     agt = agent.DQNAgent()
#     replay_memory = utils.ReplayMemory(replay_mem_size, mini_batch_size)
#     obs_history = utils.ObsHistory()
#     optimizer = optim.RMSprop(agt.qnet.parameters())
#
#     env = gym.envs.make('Pong-v4')
#
#     for episode in range(num_episodes):  # loop over episodes
#         s_t = env.reset()  # reset environment to start new episode
#         obs_history.reset(s_t)  # reset observations for new episode
#         done = False
#
#         cumulative_loss = 0
#         n_steps = 0
#         ep_r = 0
#         start_t = time.time()
#         while not done:  # loop over steps in episode
#             phi_t = obs_history.get_phi()
#             a_t = agt.act(phi_t)
#             s_t1, r_t, done, _ = env.step(a_t)
#             obs_history.store(s_t1)
#             n_steps += 1
#             ep_r += r_t
#
#             # store transition
#             replay_memory.store(s_t, a_t, r_t, done)
#             s_t = s_t1
#
#             # perform a mini-batch of stochastic gradient descent
#             if replay_memory.size() > replay_memory.sample_size + 3:
#                 mini_batch = replay_memory.sample()
#
#                 optimizer.zero_grad()
#                 loss = train.mini_batch_loss(mini_batch, agt)
#                 loss.backward()
#                 optimizer.step()
#                 cumulative_loss += loss.item()
#
#         end_t = time.time()
#         dt = int(end_t - start_t)
#         print('Episode #{}. Avg loss: {}. Episode reward: {}. Duration: {}s'
#               .format(episode, cumulative_loss / n_steps, ep_r, dt))
#         if episode % 10 == 9:
#             torch.save(agt.qnet.state_dict(), 'dqn_agt.pt')