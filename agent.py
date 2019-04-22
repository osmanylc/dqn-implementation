from collections import deque

import gym
import dqn
import utils
import train

class DQNAgent:
    
    def __init__(self):
        self.s_seq = []
        self.phi_seq = []
        self.replay_mem = ReplayMemory()

        self.q_net = QNet()
        self.epsilon = .9
        
        self.env = gym.envs.make('PongNoFrameskip-v4')
        
    
    
    def act(self):
        pass
    
    def reset(self):
        pass
    
    def _train(self):
        pass
    
    def _update_epsilon(self):
        pass



class ReplayMemory:
    
    def __init__(self, N, sample_size):
        self.N = N
        self.sample_size = sample_size
        self.transitions = deque()
        
    def sample(self, k=None):
        if k is None:
            k = self.sample_size
        return np.random.choice(self.transitions, k)

    def add(e):
        if len(self.transitions >= self.N):
            self.transitions.popleft()
        self.transitions.append(e)
 
