#!/usr/bin/env python3

"""
train.py

This script trains an RL agent on one of the custom environments.

Usage:
TODO
"""

from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from tqdm import tqdm

from datingrl.env import RealScoreEnv

config = (
	PPOConfig()
	.api_stack(
		enable_rl_module_and_learner=True,
		enable_env_runner_and_connector_v2=True,
	)
	.environment(RealScoreEnv, env_config={})
	.env_runners(num_env_runners=1)
)

algo = config.build()

for i in tqdm(range(1, 41)):

	algo.train()

	if i % 10 == 0:

		checkpoint_dir = algo.save_to_path((Path('checkpoints') / f'real_score_{i}').resolve().as_uri())
		print(f"Checkpoint saved in directory {checkpoint_dir}")
