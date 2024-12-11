"""
TODO
"""

import numpy as np

from datingrl.env import RealScoreEnv

env = RealScoreEnv()

print(env.observation_space)
print(env.action_space)

env.reset(0)
assert env.ranks.min() == 0
assert env.ranks.max() == 999

print('Testing reject all:')
print(env.reset(0))

for _ in range(1000):

	print(env.step(0))

print('Testing commits:')
for i in range(10):

	env.reset(0)

	for _ in range((i + 1) * 100 - 1):

		observation, _, terminated, _, _ = env.step(0)
		assert not terminated

	print(env.step(1))
	print(np.sum(env.candidates < observation[1]))
