"""
test_env.py

This script outputs states from an environment to allow a qualitative test.
"""

import numpy as np

from datingrl.env import RealScoreEnv

env = RealScoreEnv({})

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

		observation, _, terminated, _, _ = env.step(0)
		assert not terminated

	obs, *rest = env.step(1)
	# the observation's "value" should match with the jth reject printed above
	print(i * 100 - 1, *obs, rest)
	print(np.sum(env.scores < observation[1]))
