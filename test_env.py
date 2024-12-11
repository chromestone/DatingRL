"""
TODO
"""

from datingrl.env import RealScoreEnv

env = RealScoreEnv()

print(env.action_space)

env.reset(0)

for _ in range(1000):

	print(env.step(0))
