#!/usr/bin/env python
# coding: utf-8

# # TODO
# 
# - [x] Implement action repetition every 4 frames.
# - [x] Change model architecture
# - [x] Huber loss in gradient step.
# - [x] Save model and optimizer.
# - [x] Time training.
# - [x] Use tricks from paper (data collection, rmsprop)
# - [x] Optimize batch loss calc
# - [ ] Save statistics from paper.
# - [ ] Pick out frames with obvious Q vals and graph them

# In[ ]:


# ## Imports

# In[ ]:


from collections import deque
import random
import time

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
import numpy as np

import gym


# In[ ]:


random.seed(0)


# ## Object Definitions

# In[ ]:


def process_frame(frame):
    """
    Turn game frame into small, square, grayscale image.
    """
    pipeline = transforms.Compose([
        transforms.ToPILImage(),  # turn numpy ndarray into PIL image
        transforms.Grayscale(),  # convert image to grayscale
        transforms.Resize((84,84)),  # resize image to 84 x 84
        transforms.ToTensor()  # convert PIL image to torch tensor
    ])

    return pipeline(frame)


# In[ ]:


class QNet(nn.Module):

    def __init__(self, num_actions):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32,
                               kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64,
                               kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64,
                               kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view((x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# In[ ]:


class DQNAgent:
    
    def __init__(self):
        """
        Create an agent that uses DQN to guide its policy.

        This agent contains:
            - A history of the states it has been in, for its current episode.
            - A history of processed states for its current episode.
            - The recent transitions it has made.
            - The epsilon greedy strategy it's using.
         """
        self.env = gym.envs.make('PongNoFrameskip-v4')

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.qnet = QNet(self.env.action_space.n).to(device=self.device)
        self.target = QNet(self.env.action_space.n).to(device=self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.target.eval()
        self.optimizer = optim.Adam(self.qnet.parameters())

        self.epsilon = 1
        self.annealing_steps = int(1e6)
        self.min_epsilon = .1
        self.step_size = (self.epsilon - self.min_epsilon) / self.annealing_steps

    def act(self, phi):
        # select action using epsilon greedy strategy
        u = random.random()
        if u < self.epsilon:  # with probability epsilon, select action uniformly at random
            a = random.randrange(self.env.action_space.n)
        else:  # otherwise, select best action
            phi = phi.unsqueeze(0).to(self.device)
            with torch.no_grad():
                a = self.qnet(phi).argmax(1)

        self._update_epsilon()
        return a
    
    def _update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.step_size


# In[ ]:


class ReplayMemory:

    def __init__(self, n, sample_size):
        self.n = n
        self.sample_size = sample_size

        self.xs = torch.empty((self.n, 84, 84), dtype=torch.float)
        self.actions = torch.empty(self.n, dtype=torch.long)
        self.rewards = torch.empty(self.n, dtype=torch.float)
        self.dones = torch.empty(self.n, dtype=torch.uint8)
        self.idx = 0
        self.size = 0


    def sample(self, k=None):
        if k is None:
            k = self.sample_size

        # start at 3 because we need 4 frames
        # end at mem_size-1 because we need phi_t1
        idxs = torch.randint(3, self.size-1, (k,))

        phi, phi_1 = self.get_phis(idxs)
        return (phi,
                self.actions.index_select(0, idxs),
                self.rewards.index_select(0, idxs),
                phi_1,
                self.dones.index_select(0,idxs))

    def store(self, s, a, r, done):
        x = process_frame(s)

        self.xs[self.idx] = x
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.dones[self.idx] = int(done)

        self.idx  = (self.idx + 1) % self.n
        self.size = min(self.size + 1, self.n)

    def get_phi(self, i):
        return self.xs[i-3:i+1]

    def get_phis(self, idxs):
        phi_t = []
        phi_t1 = []

        for i in idxs:
            phi_t.append(self.get_phi(i))
            phi_t1.append(self.get_phi(i+1))

        return torch.stack(phi_t), torch.stack(phi_t1)


    def get_transition(self, t):
        phi_t = self.get_phi(t)
        phi_t1 = self.get_phi(t + 1)


        return (phi_t, self.actions[t], self.rewards[t], phi_t1, self.dones[t])


# In[ ]:


class ObsHistory:

    def __init__(self):
        self.obs4 = None
        self.phi = None

    def reset(self, obs_init):
        obs_init_p = process_frame(obs_init)
        self.obs4 = deque(4 * [obs_init_p])
        self.phi = torch.cat(tuple(self.obs4))

    def store(self, obs):
        obs = process_frame(obs)
        self.obs4.append(obs)
        self.obs4.popleft()

        self.phi = torch.cat(tuple(self.obs4))

    def get_phi(self):
        return self.phi


# ## Explore

# ### Collect 10 random frames

# In[ ]:


def initialize(replay_mem_size, batch_size):
    agt = DQNAgent()
    replay_mem = ReplayMemory(replay_mem_size, batch_size)
    obs_history = ObsHistory()
    env = gym.envs.make('PongNoFrameskip-v4')
    train_stats = TrainingStats(agt)
    
    return agt, replay_mem, obs_history, env, train_stats


def mini_batch_to_tensor(mini_batch, agt):
    phi, a, r, phi_1, dones = mini_batch
    
    phi = phi.to(agt.device, non_blocking=True)
    phi_1 = phi.to(agt.device, non_blocking=True).detach()
    a = a.to(agt.device, torch.long, non_blocking=True)
    r = r.to(agt.device, torch.float, non_blocking=True)
    dones = dones.to(agt.device, torch.float, non_blocking=True)
    
    return phi, a, r, phi_1, dones
    
    
def mini_batch_loss(mini_batch, gamma, agt):
    phi, a, r, phi_1, dones = mini_batch_to_tensor(mini_batch, agt)
    
    q_phi_1 = agt.target(phi_1).max(1)[0] * (1 - dones)
    y = (r + gamma * q_phi_1).detach()
    q_trans = agt.qnet(phi).gather(1, a.unsqueeze(1))

    loss = nn.SmoothL1Loss()
    return loss(y, q_trans)


def gradient_step(replay_mem, agt, gamma):
    if replay_mem.size > replay_mem.sample_size + 3:
        mini_batch = replay_mem.sample()
        
        agt.optimizer.zero_grad()
        loss = mini_batch_loss(mini_batch, gamma, agt)
        loss.backward()
        agt.optimizer.step()
        
        return loss.item()

    save_params(self.agt, episode_num, total_steps,
                self.ep_rewards, self.benchmark_qvals,
                self.benchmark_frames,
                'dqn_agt_{}.pt'.format(episode_num))

def save_params(agt, episodes, total_steps,
                ep_rewards, benchmark_qvals,
                benchmark_frames, save_path):
    torch.save({
        'model_state_dict': agt.qnet.state_dict(),
        'optimizer_state_dict': agt.optimizer.state_dict(),
        'episodes': episodes,
        'total_steps': total_steps,
        'ep_rewards': ep_rewards,
        'benchmark_qvals': benchmark_qvals,
        'benchmark_frames': benchmark_frames
    }, save_path)

    
def load_params(load_path):
    checkpoint = torch.load(checkpoint_path)
    
    agt.qnet.load_state_dict(checkpoint['model_state_dict'])
    agt.opimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return chekpoint['episodes'], checkpoint['total_steps']


def reset_episode(env, obs_history):
    s_t = env.reset()
    obs_history.reset(s_t)
    done = False
    
    return s_t, done


# In[ ]:


def get_rand_transitions(k, n):
    env = gym.envs.make('PongNoFrameskip-v4')
    s_t = env.reset()
    
    replay_mem = ReplayMemory(n, k)
    
    for i in range(n):
        a_t = random.randrange(env.action_space.n)
        s_t1, r_t, done, _ = env.step(a_t)
        replay_mem.store(s_t, a_t, r_t, done)
        
        s_t = s_t1
        
        if done:
            s_t = env.reset()
            
    return replay_mem.sample()


# In[ ]:


def frames_to_phi(frames):
    frames = [process_frame(s) for s in frames]
    phi = torch.cat(frames)

    return phi

def get_rand_phis(k, n):
    frames = []
    env = gym.envs.make('PongNoFrameskip-v4')
    env.reset()

    for i in range(n):
        a = random.randrange(env.action_space.n)
        s_t1, r_t, done, _ = env.step(a)
        frames.append(s_t1)

        if done:
            s_t = env.reset()

    idxs = random.sample(range(3, n), k)
    phis = [frames_to_phi(frames[i-3:i+1]) for i in idxs]

    return phis


# In[ ]:


class TrainingStats:
    
    def __init__(self, agt):
        self.agt = agt
        self.total_steps = 0
        self.ep_rewards = []
        self.steps_per_ep = []
        self.benchmark_qvals = []
        self.benchmark_frames =         torch.stack(get_rand_phis(10, 10000)).to(agt.device)
        self.t = time.time()

    def store(self, ep_reward, ep_steps, episode_num, total_steps):
        self.ep_rewards.append(ep_reward)
        self.steps_per_ep.append(ep_steps)
        self.total_steps += ep_steps

        avg_qvals = self.get_frames_avg_qval()
        self.benchmark_qvals.append(avg_qvals)
        ep_dur = time.time() - self.t
        self.t = time.time()

        if episode_num % 500 == 0:
            save_params(self.agt, episode_num, total_steps,
                        self.ep_rewards, self.benchmark_qvals,
                        self.benchmark_frames,
                        'adam_player/dqn_agt_{}.pt'.format(episode_num))

        print('Episode {}:'.format(episode_num))
        print('Reward: {}'.format(ep_reward))
        print('Total steps: {}'.format(self.total_steps))
        print('Avg qvals: {:.5f}'.format(avg_qvals))
        print('Duration: {:.2f}'.format(ep_dur))
        print('===========================================')
        

    def get_frames_avg_qval(self):
        qvals = self.agt.target(self.benchmark_frames).max(1)[0]

        return torch.mean(qvals).item()


# In[ ]:


gamma = .99
replay_mem_size = int(4e5)
batch_size = 32
num_episodes = int(2e4)

ep_rewards = []
ep_avg_train_losses = []
steps_per_ep = []
benchmark_qvals = []


# In[ ]:


torch.cuda.is_available()


# In[ ]:


agt, replay_mem, obs_history, env, train_stats =     initialize(replay_mem_size, batch_size)

total_steps = 0

for episode in range(num_episodes):
    s_t, done = reset_episode(env, obs_history)
    a_t = 0
    
    ep_reward = 0
    ep_steps = 0
    
    while not done:
        if ep_steps % 4 == 0: # select action every 4 frames
            phi_t = obs_history.get_phi()
            a_t = agt.act(phi_t)
        s_t1, r_t, done, _ = env.step(a_t)
        
        obs_history.store(s_t1)
        replay_mem.store(s_t, a_t, r_t, done)
        s_t = s_t1
        
        if total_steps > 50000:
            loss_val = gradient_step(replay_mem, agt, gamma)
        
        if total_steps % 1000 == 0:
            agt.target.load_state_dict(agt.qnet.state_dict())
            

        ep_reward += r_t
        ep_steps += 1
        total_steps += 1
        
    
    train_stats.store(ep_reward, ep_steps, episode, 
                      total_steps)
