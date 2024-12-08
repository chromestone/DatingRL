"""
TODO
"""

from typing import Optional

import gymnasium as gym
import numpy as np

class RealScoreEnv(gym.Env):
    """
    This environment only outputs the index with the real score as the observation.
    """

    def __init__(self, n: int = 1000):
        """
        TODO
        """

        self.n = n

        self.observation_space = gym.spaces.Box(
            low=  np.array([0.0, -np.inf], dtype=np.float32),
            high= np.array([1.0,  np.inf], dtype=np.float32),
            dtype=np.float32)

        self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)

        self.candidates = self.np_random.standard_normal(self.n, dtype=np.float32) * 15.0 + 50.0
        orders = self.candidates.argsort()
        self.ranks = orders.argsort()

        self.candidate_idx = 0

        observation = np.array([0.0, self.candidates[self.candidate_idx]], dtype=np.float32)
        info = {}

        return observation, info

    def step(self, action: int):

        if action == 0:

            self.candidate_idx += 1

            if self.candidate_idx < self.n:

                observation = np.array([self.candidate_idx / self.n, self.candidates[self.candidate_idx]], dtype=np.float32)
                reward = 0.0
                terminated = False

            # we have rejected all candidates and will forever be single :(
            else:

                observation = np.array([1.0, np.nan], dtype=np.float32)
                reward = -50.0
                terminated = True

        else:

            observation = np.array([1.0, np.nan], dtype=np.float32)
            # +1 because we are 0 indexed and we give some > 0 reward for any candidate (even the worst)
            reward = self.ranks[self.candidate_idx] + 1
            terminated = True

        truncated = False
        info = {}

        return observation, float(reward), terminated, truncated, info
