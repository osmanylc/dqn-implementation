import sys
sys.path.append('.')

import random
import gym

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

# set random seed
random.seed(0)


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
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.qnet = QNet(6).to(device=self.device)
        self.optimizer = optim.RMSprop(self.qnet.parameters())

        self.epsilon = .9
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
            a = self.get_best_actions(phi)

        self._update_epsilon()
        return a

    def get_best_actions(self, x):
        return self.qnet(x.to(self.device)).argmax(1)

    def get_best_values(self, x):
        return self.qnet(x.to(self.device)).max(1)[0]
    
    def _update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.step_size


class QNet(nn.Module):

    def __init__(self, num_actions):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 16,
                               kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32,
                               kernel_size=4, stride=2)
        self.fc1 = nn.Linear(2592, 256)
        self.fc2 = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((x.shape[0], -1))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
