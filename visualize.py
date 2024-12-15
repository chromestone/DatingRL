#!/usr/bin/env python3

"""
visualize.py
TODO
"""

from pathlib import Path

import numpy as np
from PIL import Image as im

from ray.rllib.algorithms.algorithm import Algorithm

algo = Algorithm.from_checkpoint((Path('checkpoints') / 'real_score_10').resolve().as_uri())
l = algo.learner_group._learner._module
p = algo.get_policy()

img_arr = np.empty((100, 100), dtype=np.uint8)

for i in range(100):

	idx = (100 - i) / 100

	for j in range(100):

		zscore = (j - 50) / 15

		obs = np.array([[idx, zscore]], dtype=np.float32)

		action = p.compute_single_action(obs)

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
data.save('visual3.png')
