#!/usr/bin/env python3

"""
visualize_rank.py
TODO
"""

import numpy as np
from PIL import Image as im

from datingrl.optimal_model import get_action_by_rank

img_arr = np.full((400, 400), 127, dtype=np.uint8)

for i in range(100):

	candidates_remaining = (100 - i) / 100

	for j in range(i + 1):

		running_rank = (j + 1) / (i + 1)

		action = get_action_by_rank(100, candidates_remaining, running_rank)

		img_arr[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4] = action * 255

data = im.fromarray(img_arr, mode='L')
data.save('visualize_rank.png')
