"""
TODO
"""

import math

INVERSE_E = 1 / math.e
REJECT = 0
COMMIT = 1

class OptimalModel:
	"""
	According to Wikipedia:
	The optimal stopping rule prescribes always rejecting the first
	∼ n / e {\displaystyle \sim n/e} applicants that are interviewed
	and then stopping at the first applicant who is better than every
	applicant interviewed so far (or continuing to the last applicant
	if this never occurs).
	"""

	def __init__(self, real_scores=True):

		self.real_scores = real_scores
		self.best_obs = None

	def get_action(self, obs) -> int:

		candidates_remaining = obs[0]
		candidates_seen = 1 - candidates_remaining
		score = obs[1]

		if real_scores:

			if candidates_seen > INVERSE_E:

				if score >= self.best_obs or candidates_remaining <= 1 / 1000:

					return COMMIT

				else:

					return REJECT 

			else:

				if self.best_obs is None or score > self.best_obs:

					self.best_obs = score

				return REJECT

		# caller keeps track of rank. this is the easiest case to handle
		else:

			if candidates_seen > INVERSE_E:

				if score == 1 or candidates_remaining <= 1 / 1000:

					return COMMIT

				else:

					return REJECT

			else:

				return REJECT
