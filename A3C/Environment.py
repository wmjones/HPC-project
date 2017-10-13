from Queue import Queue
import numpy as np
import scipy.misc as misc

class Environment:
    def __init__(self):
        self.previous_state = None
        self.current_state = None
        self.total_reward = 0

    def _get_current_state(self):  # will need some changes
        return x_

    def reset(self):
        self.total_reward = 0
        self.previous_state = self.current_state = None

    def step(self, action):     # will need some changes
        observation = []
        reward = []
        done = []

        self.total_reward += reward
        self.previous_state = self.current_state
        self.current_state = self._get_current_state()
        return reward, done
