#!/usr/bin/env python3

"""
train.py
TODO
"""

from pprint import pprint

import ray
from ray.rllib.algorithms import ppo

from datingrl.env import RealScoreEnv

ray.init()

config = {
	'env_config': {}
}

algo = ppo.PPO(env=RealScoreEnv, config=config)

for i in range(1):

	result = algo.train()
	result.pop("config")
	pprint(result)

checkpoint_dir = algo.save_to_path()
print(f"Checkpoint saved in directory {checkpoint_dir}")
