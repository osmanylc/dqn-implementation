import random
from collections import deque

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
        transforms.Resize((110, 84)),  # resize image to 110 x 84
        transforms.CenterCrop(84),  # crop at the center into 84 x 84 image
        transforms.ToTensor()  # convert PIL image to torch tensor
    ])

    return pipeline(frame)


class ReplayMemory:

    def __init__(self, n, sample_size):
        self.sample_size = sample_size
        self.transitions = []
        self.idx = 0
        self.n = n

    def sample(self, k=None):
        if k is None:
            k = self.sample_size

        mem_size = self.size()

        # start at 3 because we need 4 frames
        # end at mem_size-1 because we need phi_t1
        transition_idxs = random.sample(range(3, mem_size-1), k)

        return [self.get_transition(i) for i in transition_idxs]

    def store(self, s, a, r, done):
        if len(self.transitions) < self.n:
            self.transitions.append(None)

        x = process_frame(s)
        self.transitions[self.idx] = (x, a, r, done)

        self.idx  = (self.idx + 1) % self.n

    def size(self):
        return len(self.transitions)

    def get_phi(self, i):
        assert i >= 3

        frames = [self.transitions[i - j][0]
                  for j in [3, 2, 1, 0]]
        phi = torch.cat(frames)

        return phi

    def get_transition(self, t):
        phi_t = self.get_phi(t)
        phi_t1 = self.get_phi(t + 1)

        _, a_t, r_t, done = self.transitions[t]

        return (phi_t, a_t, r_t, phi_t1, done)


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