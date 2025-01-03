"""
stateful.py

This module contains stateful agents.
Users of this module should take care to associate one environment or one run of an environment
with a single instance of an agent from this module. Once the environment is terminated or
truncated, you must create a new instance or else the behavior of compute_single_action is
undefined.
"""

import numpy as np

from ..constants import REJECT, COMMIT

class OptimalAgent:
	"""
	An agent that takes actions according to the optimal policy under classical secretary problem
	assumptions.
	"""

	INVERSE_E = 1 / np.e

	def __init__(self, n: int):

		self.n = n
		self.best_obs = None

	def compute_single_action(self, observation: tuple[float, float]) -> int:

		# This is the proportion of candidates that have been rejected.
		# In the past, I have called this "candidates_seen".
		# However, we do not consider the current candidate on which an action is tbd as "seen".
		candidates_rejected = 1 - observation[0]
		score = observation[1]

		if candidates_rejected >= OptimalAgent.INVERSE_E:

			if score >= self.best_obs or candidates_rejected >= (self.n - 1) / self.n:

				return COMMIT

		elif self.best_obs is None or score > self.best_obs:

			self.best_obs = score

		return REJECT

STR2AGENT = {
	'optimal': OptimalAgent
}
