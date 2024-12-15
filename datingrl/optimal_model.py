"""
TODO
"""

import math

INVERSE_E = 1 / math.e
REJECT = 0
COMMIT = 1

def get_action_by_rank(n, candidates_remaining, running_rank):
	"""
	According to Wikipedia:
	The optimal stopping rule prescribes always rejecting the first
	∼ n / e {\displaystyle \sim n/e} applicants that are interviewed
	and then stopping at the first applicant who is better than every
	applicant interviewed so far (or continuing to the last applicant
	if this never occurs).
	"""

	candidates_seen = 1 - candidates_remaining

	if candidates_seen > INVERSE_E:

		if running_rank > (n - 1) / n or candidates_seen >= (n - 1) / n:

			return COMMIT

		else:

			return REJECT

	else:

		return REJECT

class OptimalModel:
	"""
	TODO
	"""

	def __init__(self):

		self.best_obs = None

	def get_action(self, obs) -> int:

		candidates_remaining = obs[0]
		candidates_seen = 1 - candidates_remaining
		score = obs[1]

		if candidates_seen > INVERSE_E:

			if score >= self.best_obs or candidates_remaining <= 1 / 1000:

				return COMMIT

			else:

				return REJECT 

		else:

			if self.best_obs is None or score > self.best_obs:

				self.best_obs = score

			return REJECT
