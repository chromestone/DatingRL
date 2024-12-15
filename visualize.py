#!/usr/bin/env python3

"""
visualize.py
TODO
"""

from pathlib import Path

import numpy as np

from PIL import Image

from ray.rllib.core.rl_module import RLModule

import torch

# Create only the neural network (RLModule) from our checkpoint.
torch_rl_module = RLModule.from_checkpoint(
    (Path('checkpoints') / 'real_score_30' / 'learner_group' / 'learner' / 'rl_module').resolve().as_uri()
)["default_policy"]

def compute_actions(obs: np.ndarray[np.float32]) -> np.ndarray[np.int64]:
	"""
	Compute actions from a batch of observations.

	Args:
		obs: A numpy array of shape (batch, *observation_shape).

	Returns:
		Numpy array of shape (batch,) containing predicted actions.
	"""

	# Compute the next action from a batch of observations.
	torch_obs_batch = torch.from_numpy(obs)

	action_logits = torch_rl_module.forward_inference({
		'obs': torch_obs_batch
	})['action_dist_inputs']

	# The default RLModule used here produces action logits (from which
	# we'll have to sample an action or use the max-likelihood one).
	return torch.argmax(action_logits, dim=1).numpy()

# Create the 100x100x2 array
rows, cols = 100, 100
obs = np.empty((rows, cols, 2), dtype=np.float32)

# Compute idx for all rows
idx = (100 - np.arange(rows)) / 100

# Compute zscore for all columns
zscore = (np.arange(cols) - 50) / 15

# Use broadcasting to combine idx and zscore into obs
obs[:, :, 0] = idx[:, np.newaxis]
obs[:, :, 1] = zscore[np.newaxis, :]

actions = compute_actions(obs.reshape((rows * cols, 2))).reshape(rows, cols)
assert np.all((actions == 0) | (actions == 1))

img_arr = actions * 255

print(np.unique(img_arr, return_counts=True))

data = Image.fromarray(np.uint8(img_arr), mode='L')
data.save('visual.png')
