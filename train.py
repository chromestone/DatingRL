#!/usr/bin/env python3

"""
train.py
TODO
"""

import ray
from ray.rllib.algorithms import ppo

from datingrl.env import RealScoreEnv

ray.init()

config = {
	'env_config': {}
}

algo = ppo.PPO(env=RealScoreEnv, config=config)

for i in range(10):

	print(i)
	algo.train()

checkpoint_dir = algo.save_to_path()
print(f"Checkpoint saved in directory {checkpoint_dir}")
