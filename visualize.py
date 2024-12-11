#!/usr/bin/env python3

"""
train.py
TODO
"""

import numpy as np
from PIL import Image as im
from stable_baselines3 import PPO
from datingrl.env import RealScoreEnv

env = RealScoreEnv()

print(env.action_space)

model = PPO("MlpPolicy", env, verbose=1)
model.load('realscoreenv_ppo_mlp.zip')

img_arr = np.empty((400, 400), dtype=np.uint8)

for i in range(100):

    idx = i / 100

    for j in range(100):

        score = j

        obs = np.array([idx, score], dtype=np.float32)
        action, _ = model.predict(obs, deterministic=True)

        img_arr[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4] = action * 255

print(np.unique(img_arr, return_counts=True))
data = im.fromarray(img_arr, mode='L')
data.save('test.jpg')
