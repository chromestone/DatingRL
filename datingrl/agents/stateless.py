"""
stateless.py

This module contains stateless agents. Statelessness refers to the fact that past observations are
not tracked.
For any instance of an agent from this module, its compute_actions function should be deterministic
and always return the same action for the same observation.
"""

from typing import Optional

import numpy as np

from ..constants import REJECT, COMMIT

class OptimalAgent:
	"""
	An agent that takes actions according to the optimal policy under classical secretary problem
	assumptions.

	This agent expects the observation space to be of 2 values.
	The first value is the proportion of candidates remaining. The first value is within (0, 1].
	The second value is the running rank divided by the number of candidates. The best candidate
	shall have a running rank of n, where n is the number of candidates. Thus, the second value
	shall be within (0, 1].

	According to Wikipedia:
		The optimal stopping rule prescribes always rejecting the first
		~ n / e {\\displaystyle \\sim n/e} applicants that are interviewed
		and then stopping at the first applicant who is better than every
		applicant interviewed so far (or continuing to the last applicant
		if this never occurs).
	"""

	INVERSE_E = 1 / np.e

	def __init__(self, n: int):
		"""
		Initializes an OptimalAgent instance.

		Args:
			n: The number of candidates.
		"""

		self.n = n

	def compute_actions(self, observations: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
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
		candidates_rejected = 1 - observations[:, 0]
		running_rank = observations[:, 1]

		actions = np.empty((observations.shape[0], ))
		probs = None

		can_select = candidates_rejected >= OptimalAgent.INVERSE_E

		actions[can_select] = np.where(
			# criteria for COMMIT
			(
				(running_rank       [can_select] >  (self.n - 1) / self.n) |
				(candidates_rejected[can_select] >= (self.n - 1) / self.n)
			),
			COMMIT,
			# else REJECT
			REJECT
		)
		actions[~can_select] = REJECT

		return actions, probs

STR2AGENT = {
	'optimal': OptimalAgent
}