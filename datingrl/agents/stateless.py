"""
stateless.py

This module contains stateless agents. Statelessness refers to the fact that past observations are
not tracked.
For any instance of an agent from this module, its compute_actions function should be deterministic
and always return the same action for the same observation.
"""

from pathlib import Path

import numpy as np

from ray.rllib.core.rl_module import RLModule

import torch

from ..constants import REJECT, COMMIT, INVERSE_E

from ..types import ACTIONS_PROBS_TUPLE, INT_FLOAT_TUPLE

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

	def __init__(self, n: int):
		"""
		Initializes an OptimalAgent instance.

		Args:
			n: The number of candidates.
		"""

		self.n = n

	def compute_actions(self, observations: np.ndarray[np.float32]) -> ACTIONS_PROBS_TUPLE:
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

		actions = np.empty((observations.shape[0], ), dtype=np.float32)

		can_select = candidates_rejected >= INVERSE_E

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

		return actions, None

	def compute_single_action(self, observation: tuple[float, float]) -> INT_FLOAT_TUPLE:
		"""
		Compute a single action on a single observation.
		See the class documentation above for expected values in the observation.

		Args:
			observation: A single observation tuple.

		Returns:
			A tuple of an action and its corresponding probability (None in this case)
		"""

		actions, _ = self.compute_actions(np.array([observation], dtype=np.float32))
		return actions[0], None

class PPOAgent:
	"""
	A deep reinforcement learning agent.

	This agent loads the 'learner_group/learner/rl_module' RLlib RLModule from a checkpoint and
	uses it to compute actions.

	See the OptimalAgent class documentation for expected values in the observations.
	"""

	def __init__(self, n: int, checkpoint_path: str):
		"""
		Initializes a PPOAgent instance.

		Args:
			n: The number of candidates.
			checkpoint_path: The path to the RLlib saved checkpoint.
		"""

		self.n = n

		# Create only the neural network (RLModule) from our checkpoint.
		self.torch_rl_module = RLModule.from_checkpoint(
			(Path(checkpoint_path) / 'learner_group' / 'learner' / 'rl_module').resolve().as_uri()
		)["default_policy"]

	def compute_actions(self, observations: np.ndarray[np.float32]) -> ACTIONS_PROBS_TUPLE:
		"""
		Compute actions for a batch of observations.
		See the class documentation above for expected values in the observations.

		Args:
			observations: A batch of observations of shape (batch, 2).

		Returns:
			An array of actions of shape (batch, ) and probabilities of shape (batch, )
		"""

		actions = np.empty((observations.shape[0], ), dtype=np.float32)
		probs = np.empty((observations.shape[0], ), dtype=np.float32)

		candidates_rejected = 1 - observations[:, 0]
		# at the last candidate we are forced to COMMIT
		forced_choice = candidates_rejected >= (self.n - 1) / self.n

		actions[forced_choice] = COMMIT
		# we assume that choosing the last candidate is better than being alone forever
		# thus we can confidently set the probability to 1 here
		probs[forced_choice] = 1.0

		# the model will predict on everything else
		np_obs_batch = observations[~forced_choice]
		if np_obs_batch.shape[0] > 0:

			# Compute the next action from a batch of observations.
			torch_obs_batch = torch.from_numpy(np_obs_batch)

			action_logits = self.torch_rl_module.forward_inference({
				'obs': torch_obs_batch
			})['action_dist_inputs']

			# assign actions and probs back into their original positions
			# The default RLModule used here produces action logits (from which
			# we'll have to sample an action or use the max-likelihood one).
			actions[~forced_choice] = torch.argmax(action_logits, dim=1).numpy()
			probs[~forced_choice] = torch.softmax(action_logits, dim=1).numpy()[:, COMMIT]

		return actions, probs

	def compute_single_action(self, observation: tuple[float, float]) -> INT_FLOAT_TUPLE:
		"""
		Compute a single action on a single observation.
		See the class documentation above for expected values in the observation.

		Args:
			observation: A single observation tuple.

		Returns:
			A tuple of an action and its corresponding probability
		"""

		actions, probs = self.compute_actions(np.array([observation], dtype=np.float32))
		return actions[0], probs[0]

class DQNAgent:
	"""
	A deep reinforcement learning agent.

	This agent loads the 'learner_group/learner/rl_module' RLlib RLModule from a checkpoint and
	uses it to compute actions.

	See the OptimalAgent class documentation for expected values in the observations.
	"""

	def __init__(self, n: int, checkpoint_path: str):
		"""
		Initializes a DQNAgent instance.

		Args:
			n: The number of candidates.
			checkpoint_path: The path to the RLlib saved checkpoint.
		"""

		self.n = n

		# Create only the neural network (RLModule) from our checkpoint.
		self.torch_rl_module = RLModule.from_checkpoint(
			(Path(checkpoint_path) / 'learner_group' / 'learner' / 'rl_module').resolve().as_uri()
		)["default_policy"]

	def compute_actions(self, observations: np.ndarray[np.float32]) -> ACTIONS_PROBS_TUPLE:
		"""
		Compute actions for a batch of observations.
		See the class documentation above for expected values in the observations.

		Args:
			observations: A batch of observations of shape (batch, 2).

		Returns:
			An array of actions of shape (batch, ) and probabilities (None in this case)
		"""

		actions = np.empty((observations.shape[0], ), dtype=np.float32)

		candidates_rejected = 1 - observations[:, 0]
		# at the last candidate we are forced to COMMIT
		forced_choice = candidates_rejected >= (self.n - 1) / self.n

		actions[forced_choice] = COMMIT

		# the model will predict on everything else
		np_obs_batch = observations[~forced_choice]
		if np_obs_batch.shape[0] > 0:

			# Compute the next action from a batch of observations.
			torch_obs_batch = torch.from_numpy(np_obs_batch)

			torch_actions = self.torch_rl_module.forward_inference({
				'obs': torch_obs_batch
			})['actions']

			# assign actions back into their original positions
			actions[~forced_choice] = torch_actions.numpy()

		return actions, None

	def compute_single_action(self, observation: tuple[float, float]) -> INT_FLOAT_TUPLE:
		"""
		Compute a single action on a single observation.
		See the class documentation above for expected values in the observation.

		Args:
			observation: A single observation tuple.

		Returns:
			A tuple of an action and its corresponding probability (None in this case)
		"""

		actions, _ = self.compute_actions(np.array([observation], dtype=np.float32))
		return actions[0], None

STR2AGENT = {
	'optimal': OptimalAgent,
	'ppo': PPOAgent,
	'dqn': DQNAgent
}
