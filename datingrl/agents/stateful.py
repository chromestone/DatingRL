"""
stateful.py

This module contains stateful agents.
Users of this module should take care to associate one environment or one run of an environment
with a single instance of an agent from this module. Once the environment is terminated or
truncated, you must create a new instance or else the behavior of compute_single_action is
undefined.
"""

import numpy as np

from ..constants import REJECT, COMMIT, INVERSE_E

from ..types import INT_FLOAT_TUPLE

class OptimalAgent:
	"""
	An agent that takes actions according to the optimal policy under classical secretary problem
	assumptions.

	This agent expects the observation space to be of 2 values.
	The first value is the proportion of candidates remaining. The first value is within (0, 1].
	The second value is some number representing the score of candidates. This implementation
	assumes that a larger score implies a more favorable candidate.

	According to Wikipedia:
		The optimal stopping rule prescribes always rejecting the first
		~ n / e {\\displaystyle \\sim n/e} applicants that are interviewed
		and then stopping at the first applicant who is better than every
		applicant interviewed so far (or continuing to the last applicant
		if this never occurs).
	"""

	def __init__(self, n: int):
		"""
		Initializes an OptimalAgent instance.

		Args:
			n: The number of candidates.
		"""

		self.n = n
		self.best_obs = None

	def compute_single_action(self, observation: tuple[float, float]) -> INT_FLOAT_TUPLE:
		"""
		Compute actions for a batch of observations.
		See the class documentation above for expected values in the observations.

		Args:
			observations: A batch of observations of shape (batch, 2).

		Returns:
			An array of actions of shape (batch, ) and probabilities (None in this case)
		"""

		# This is the proportion of candidates that have been rejected.
		# In the past, I have called this "candidates_seen".
		# However, we do not consider the current candidate on which an action is tbd as "seen".
		candidates_rejected = 1 - observation[0]
		score = observation[1]

		if candidates_rejected >= INVERSE_E:

			if self.best_obs is None:

				raise RuntimeError(
					f'compute_single_action received candidates remaining {observation[0]} before the best observation could be set!'
				)

			if score >= self.best_obs or candidates_rejected >= (self.n - 1) / self.n:

				return COMMIT, None

		elif self.best_obs is None or score > self.best_obs:

			self.best_obs = score

		return REJECT, None

STR2AGENT = {
	'optimal': OptimalAgent
}
