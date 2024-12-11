#!/usr/bin/env python3

"""
train.py
TODO
"""

from stable_baselines3 import PPO

from datingrl.env import RealScoreEnv

env = RealScoreEnv()

print(env.action_space)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)

model.save('realscoreenv_ppo_mlp')

env.close()

