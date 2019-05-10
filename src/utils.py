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
        self.n = n
        self.sample_size = sample_size

        self.xs = np.empty((self.n, 84, 84), dtype=np.uint8)
        self.actions = np.empty(self.n, dtype=np.uint8)
        self.rewards = np.empty(self.n, dtype=np.int8)
        self.dones = np.empty(self.n, dtype=np.bool)
        self.idx = 0
        self.size = 0


    def sample(self, k=None):
        if k is None:
            k = self.sample_size

        # start at 3 because we need 4 frames
        # end at mem_size-1 because we need phi_t1
        transition_idxs = random.sample(range(3, self.size-1), k)

        return [self.get_transition(i) for i in transition_idxs]

    def store(self, s, a, r, done):
        x = process_frame(s)

        self.xs[self.idx] = x
        self.actions[self.idx] = a
        self.rewards[self.idx] = r
        self.dones[self.idx] = done

        self.idx  = (self.idx + 1) % self.n
        self.size = min(self.size + 1, self.n)

    def get_phi(self, i):
        assert i >= 3

        frames = [torch.tensor(self.xs[i - j], dtype=torch.float)
                  for j in [3, 2, 1, 0]]
        phi = torch.stack(frames)

        return phi

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