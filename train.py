#!/usr/bin/env python3

"""
train.py

This script trains an RL agent on one of the custom environments.

Usage:
TODO
"""

from argparse import ArgumentParser
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig

from tqdm import tqdm

from datingrl.envs import STR2ENV

parser = ArgumentParser(description='TODO')

parser.add_argument('env', choices=STR2ENV.keys(), help='Name of an environment in datingrl.env')

args = parser.parse_args()

env_class = STR2ENV[args.env]

config = (
	PPOConfig()
	.api_stack(
		enable_rl_module_and_learner=True,
		enable_env_runner_and_connector_v2=True,
	)
	.environment(env_class, env_config={})
	.env_runners(num_env_runners=1)
)

algo = config.build()

for i in tqdm(range(1, 41)):

	algo.train()

	if i % 10 == 0:

		checkpoint_dir = algo.save_to_path((Path('checkpoints') / f'{args.env}_{i}').resolve().as_uri())
		print(f"Checkpoint saved in directory {checkpoint_dir}")
