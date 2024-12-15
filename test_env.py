"""
test_env.py

This script outputs states from an environment to allow a qualitative test.
"""

from argparse import ArgumentParser

import numpy as np

from datingrl.env import STR2ENV

parser = ArgumentParser(description='Output some states from an environment')

parser.add_argument('env', choices=STR2ENV.keys(), help='Name of an environment in datingrl.env')

args = parser.parse_args()

env_class = STR2ENV[args.env]
env = env_class({})

print('Observation Space:')
print(env.observation_space)
print('\nAction Space:')
print(env.action_space)

env.reset(0)
assert env.ranks.min() == 0
assert env.ranks.max() == 999

print('\nTesting reject all:')
obs, info = env.reset(0)
print(0, *obs, info)

for i in range(1, 1001):

	obs, *rest = env.step(0)
	print(i, *obs, rest)

print('\nTesting commits:')
for i in range(1, 11):

	env.reset(0)

	# -1 so we can commit on the last step
	for _ in range(i * 100 - 1):

		_, _, terminated, _, _ = env.step(0)
		assert not terminated

	obs, *rest = env.step(1)
	# the observation's "value" should match with the jth reject printed above
	print(i * 100 - 1, *obs, rest)

	if args.env == 'real_score':

		print('\tRank:', np.sum(env.scores < obs[1]))

	elif args.env == 'running_rank':

		print('\tRank:', env.ranks[env.candidate_idx], 'Running:', env.running_ranks[env.candidate_idx])
