import sys
sys.path.append('.')

import time

import torch
import gym

import agent, utils

agt = agent.DQNAgent()
agt.qnet.load_state_dict(torch.load('dqn_agt.pt'))
agt.qnet.eval()

obs_history = utils.ObsHistory()

env = gym.envs.make('Pong-v4')
obs = env.reset()
obs_history.reset(obs)

while True:
    env.render()
    phi = obs_history.phi
    a = agt.act(phi)
    obs, r, done, _ = env.step(a)
    obs_history.store(obs)

    time.sleep(.003)
    if done:
        obs = env.reset()
