import sys
sys.path.append('.')

import gym
import torch
from torch import optim

import agent, train, utils

# hyperparameters
replay_mem_size = int(1e6)
mini_batch_size = 32
num_episodes = int(2e3)

agt = agent.DQNAgent()
replay_memory = utils.ReplayMemory(replay_mem_size, mini_batch_size)
obs_history = utils.ObsHistory()
optimizer = optim.RMSprop(agt.qnet.parameters())

env = gym.envs.make('PongNoFrameskip-v4')

for episode in range(num_episodes):  # loop over episodes
    obs_init = env.reset()  # reset environment to start new episode
    obs_history.reset(obs_init)  # reset observations for new episode
    done = False

    print('Episode #{}'.format(episode))
    if episode % 10 == 9:
        torch.save(agt.qnet.state_dict(), 'dqn_agt.pt')

    cumulative_loss = 0
    n_steps = 0
    ep_rew = 0
    while not done:  # loop over steps in episode
        phi = obs_history.phi
        a = agt.act(phi)
        obs, r, done, _ = env.step(a)
        obs_history.store(obs)
        n_steps += 1
        ep_rew += r

        # store transition
        phi_1 = obs_history.phi
        transition = utils.Transition(phi, a, r, phi_1, done)
        replay_memory.store(transition)

        # perform a mini-batch of stochastic gradient descent
        if replay_memory.size() > replay_memory.sample_size:
            mini_batch = replay_memory.sample()

            optimizer.zero_grad()
            loss = train.mini_batch_loss(mini_batch, agt)
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.item()
    print('Avg episode batch loss: {}'.format(cumulative_loss / n_steps))
    print('# steps: {}'.format(n_steps))
    print('episode reward: {}'.format(ep_rew))
