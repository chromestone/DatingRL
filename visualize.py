#!/usr/bin/env python3

"""
train.py
TODO
"""

import numpy as np
from PIL import Image as im
from stable_baselines3 import DQN
from datingrl.env import RealScoreEnv

env = RealScoreEnv()

env.reset(0)

model = DQN("MlpPolicy", env, verbose=1)

model.load('realscore_dqn_100k.zip')

img_arr = np.empty((100, 100), dtype=np.uint8)

for i in range(100):

    idx = i / 100

    for j in range(100):

        score = j

        obs = np.array([idx, score], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)
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
