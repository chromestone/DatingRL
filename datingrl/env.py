"""
env.py

This module contains our custom Gymnasium environments.
"""

from typing import Any, Optional

import gymnasium as gym
import numpy as np

class RealScoreEnv(gym.Env):
    """
    This environment only outputs the index with the z-score as the observation.
    """

    def __init__(self, env_config: dict[str, Any]):
        """
        TODO
        """

        self.n = env_config.get('n', 1000)

        self.observation_space = gym.spaces.Box(
            low=  np.array([0.0, -np.inf], dtype=np.float32),
            high= np.array([1.0,  np.inf], dtype=np.float32),
            dtype=np.float32
        )

        self.action_space = gym.spaces.Discrete(2)

        self.terminated = True

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):

        super().reset(seed=seed)

        self.scores = self.np_random.standard_normal(self.n, dtype=np.float32)

        orders = self.scores.argsort()
        self.ranks = orders.argsort()

        self.candidate_idx = 0
        self.terminated = False

        observation = np.array([1.0, self.scores[self.candidate_idx]], dtype=np.float32)
        info = {}

        return observation, info

    def step(self, action: int):

        if self.terminated:

            raise RuntimeError('step called on terminated environment')

        if action not in (0, 1):

            raise ValueError('action must be 0 or 1')

        if action == 0:

            if self.candidate_idx < self.n - 1:

                self.candidate_idx += 1

                observation = np.array([(self.n - self.candidate_idx) / self.n, self.scores[self.candidate_idx]], dtype=np.float32)
                reward = 0.0
                terminated = False

            # we have rejected all candidates and will forever be single :(
            else:

                observation = np.array([0.0, self.scores[-1]], dtype=np.float32)
                reward = -1.0
                terminated = True

        else:

            observation = np.array([0.0, self.scores[self.candidate_idx]], dtype=np.float32)
            # +1 because we are 0 indexed and we give some > 0 reward for any candidate (even the worst)
            reward = self.ranks[self.candidate_idx] + 1
            terminated = True

        self.terminated = terminated

        truncated = False
        info = {}

        return observation, float(reward), terminated, truncated, info
