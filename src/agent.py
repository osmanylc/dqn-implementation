import sys
sys.path.append('.')

import random
import gym

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F


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
        self.env = gym.envs.make('Pong-v4')

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.qnet = QNet(self.env.action_space.n).to(device=self.device)
        self.target = QNet(self.env.action_space.n).to(device=self.device)
        self.target.load_state_dict(self.qnet.state_dict())
        self.target.eval()
        self.optimizer = optim.RMSprop(self.qnet.parameters(), lr=.00025, momentum=.95, alpha=.95, eps=.01)

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
