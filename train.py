#!/usr/bin/env python3

"""
train.py
TODO
"""

from stable_baselines3 import DQN

from datingrl.env import RealScoreEnv

env = RealScoreEnv()

model = DQN("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100_000, log_interval=1000)

model.save('realscore_dqn_100k')

env.close()
