#!/usr/bin/env python3

"""
visualize.py
TODO
"""

from pathlib import Path

import numpy as np
from PIL import Image as im

from ray.rllib.core.rl_module import RLModule

import torch

# Create only the neural network (RLModule) from our checkpoint.
rl_module = RLModule.from_checkpoint((Path('checkpoints') / "real_score_10").resolve().as_uri())["default_policy"]

img_arr = np.empty((100, 100), dtype=np.uint8)

for i in range(100):

	idx = i / 100

	for j in range(100):

		score = j

		obs = np.array([[idx, score]], dtype=np.float32)

		# Compute the next action from a batch (B=1) of observations.
		torch_obs_batch = torch.from_numpy(obs)

		action_logits = rl_module.forward_inference({"obs": torch_obs_batch})["action_dist_inputs"]

		# The default RLModule used here produces action logits (from which
		# we'll have to sample an action or use the max-likelihood one).
		action = torch.argmax(action_logits[0]).numpy()

		print(action)

		# action, _ = model.predict(obs, deterministic=True)
		#action = 0
		#for _ in range(10):

		#    a, _ = model.predict(obs, deterministic=True)
		#    action += a

		#action = np.round(action / 10)

		# img_arr[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4] = action * 255
		img_arr[i, j] = np.uint8(action * 255)

print(np.unique(img_arr, return_counts=True))
data = im.fromarray(img_arr, mode='L')
data.save('visual2.png')
