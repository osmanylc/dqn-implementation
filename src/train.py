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
            x = phi_t1.unsqueeze(0)
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

#
# def initialize(replay_mem_size, batch_size):
#     agt = agent.DQNAgent()
#     replay_mem = utils.ReplayMemory(replay_mem_size, batch_size)
#     obs_history = utils.ObsHistory()
#     env = gym.envs.make('Pong-v4')
#
#     return agt, replay_mem, obs_history, env
#
#
# def act_step(obs_history, agt, env):
#     phi_t = obs_history.get_phi()
#     a_t = agt.act(phi_t)
#     s_t1, r_t, done, _ = env.step(a_t)
#
#     return a_t, s_t1, r_t, done
#
#
# def store_step(s_t, a_t, r_t, done, s_t1, obs_history, replay_mem):
#     obs_history.store(s_t1)
#     replay_mem.store(s_t, a_t, r_t, done)
#
#
# def gradient_step(replay_memory, agt):
#     if replay_memory.size() > replay_memory.sample_size + 3:
#         mini_batch = replay_memory.sample()
#
#         agt.optimizer.zero_grad()
#         loss = mini_batch_loss(mini_batch, agt)
#         loss.backward()
#         agt.optimizer.step()
#
#
# def save_params(agt, save_path):
#     torch.save({
#         'model_state_dict': agt.qnet.state_dict(),
#         'optimizer_state_dict': agt.optimizer.state_dict()
#     }, save_path)
#
#
# def load_params(agt, load_path):
#     checkpoint = torch.load(load_path)
#
#     agt.qnet.load_state_dict(checkpoint['model_state_dict'])
#     agt.opimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#
#
# def reset_episode(env, obs_history):
#     s_t = env.reset()
#     obs_history.reset(s_t)
#     done = False
#
#     return s_t, done
#
#
# def train(replay_mem_size=int(1e6),
#           batch_size=32,
#           num_episodes=int(1e3)):
#     agt, replay_mem, obs_history, env = initialize(replay_mem_size, batch_size)
#
#     for episode in range(num_episodes):  # loop over episodes
#         s_t, done = reset_episode(env, obs_history)
#
#         while not done:  # loop over steps in episode
#             a_t, s_t1, r_t, done = act_step(obs_history, agt, env)
#             store_step(s_t, a_t, r_t, done, s_t1, obs_history, replay_mem)
#
#             s_t = s_t1
#
#             gradient_step(replay_mem, agt)
#
#         if episode % 10 == 9:
#             save_params(agt, 'dqn_agt_{}.pt'.format(episode))
