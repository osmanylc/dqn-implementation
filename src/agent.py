import random

import gym
import numpy as np

from . import dqn

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
        self.qnet = dqn.QNet(6)
        self.epsilon = .9
        self.annealing_steps = int(1e6)
        self.min_epsilon = .1
        self.step_size = (self.epsilon - self.min_epsilon) / self.annealing_steps
        
        self.env = gym.envs.make('PongNoFrameskip-v4')

    def act(self, phi):
        """

        :param phi:
        :return:
        """
        # select action using epsilon greedy strategy
        u = random.random()
        if u < self.epsilon:  # with probability epsilon, select action uniformly at random
            a = random.randrange(self.env.action_space.n)
        else:  # otherwise, select best action
            a = self.get_best_action(phi)

        self._update_epsilon()
        return a

    def get_best_action(self, phi):
        return np.argmax(self.qnet(phi))

    def _train(self):
        pass
    
    def _update_epsilon(self):
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.step_size
