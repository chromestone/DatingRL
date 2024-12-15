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

algo.train()

checkpoint_dir = algo.save_to_path((Path('checkpoints') / "real_score_10").resolve().as_uri())
print(f"Checkpoint saved in directory {checkpoint_dir}")
