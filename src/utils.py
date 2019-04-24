from collections import deque

from torchvision import transforms
import numpy as np
import torch


class ReplayMemory:

    def __init__(self, n, sample_size):
        self.n = n
        self.sample_size = sample_size
        self.transitions = deque()

    def sample(self, k=None):
        if k is None:
            k = self.sample_size
        return np.random.choice(self.transitions, k)

    def store(self, e):
        if len(self.transitions) >= self.n:
            self.transitions.popleft()
        self.transitions.append(e)


class ObsHistory:

    def __init__(self):
        self.obs4 = None
        self.phi = None

    def reset(self, obs_init):
        """

        :param obs_init:
        :return:
        """
        obs_init_p = self._process_frame(obs_init)
        self.obs4 = deque(4 * [obs_init_p])
        self.phi = torch.cat(self.obs4)

    def store(self, obs):
        """

        :param obs:
        :return:
        """
        self.obs4.append(obs)
        self.obs4.popleft()

        self.phi = torch.cat(self.obs4)

    def _process_frame(self, frame):
        """
        Turn game frame into small, square, grayscale image.
        """
        pipeline = transforms.Compose([
            transforms.ToPILImage(), # turn numpy ndarray into PIL image
            transforms.Grayscale(), # convert image to grayscale
            transforms.Resize((110, 84)), # resize image to 110 x 84
            transforms.CenterCrop(84), # crop at the center into 84 x 84 image
            transforms.ToTensor() # convert PIL image to torch tensor
        ])

        return pipeline(frame)
