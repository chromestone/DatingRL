#!/usr/bin/env python3

"""
train.py

This script trains a deep reinforcement learning (DRL) agent on one of the DatingRL environments.

Usage:
python3 train.py -a ppo -e running_rank --iters 10
python3 train.py -a dqn -e running_rank --num_ckpts 4
python3 train.py -a dqn -e running_rank --iters 200 --num_ckpts 8
"""

from argparse import ArgumentParser
from pathlib import Path

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig

from tqdm import tqdm

from datingrl.envs import STR2ENV

# supported algorithms' config
ALGORITHM2CONFIG = {
	'ppo': PPOConfig,
	'dqn': DQNConfig
}

parser = ArgumentParser(description='Train a DRL agent on of the custom DatingRL environments')

parser.add_argument(
	'-a',
	'--algorithm',
	choices=ALGORITHM2CONFIG.keys(),
	required=True,
	help='Name of the DRL algorithm to train'
)
parser.add_argument(
	'-e',
	'--environment',
	choices=STR2ENV.keys(),
	required=True,
	help='Name of an environment in datingrl.env'
)
parser.add_argument(
	'--iters',
	default=100,
	type=int,
	help='Number of times to call train'
)
parser.add_argument(
	'--num_ckpts',
	default=1,
	type=int,
	help='Number of evenly spaced checkpoints to save. Must divide iters'
)
parser.add_argument(
	'--num_runners',
	default=1,
	type=int,
	help='Number of environment runners. Sets RLlib num_env_runners'
)

args = parser.parse_args()

assert args.iters > 0
assert args.num_ckpts > 0
assert args.iters % args.num_ckpts == 0

algorithm_config = ALGORITHM2CONFIG[args.algorithm]
env_class = STR2ENV[args.environment]

config = (
	algorithm_config()
	.api_stack(
		enable_rl_module_and_learner=True,
		enable_env_runner_and_connector_v2=True,
	)
	.environment(env_class, env_config={})
	.env_runners(num_env_runners=args.num_runners)
)

algo = config.build()

iters_per_ckpt = args.iters // args.num_ckpts
for i in tqdm(range(1, args.iters + 1)):

	algo.train()

	if i % iters_per_ckpt == 0:

		checkpoint_dir = algo.save_to_path((Path('checkpoints') / f'{args.algorithm}_{args.environment}_{i}').resolve().as_uri())
		tqdm.write(f"Checkpoint saved in directory {checkpoint_dir}")
