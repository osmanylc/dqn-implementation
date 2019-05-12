import random
from collections import deque
import time

import torch
import numpy as np
from torchvision import transforms


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