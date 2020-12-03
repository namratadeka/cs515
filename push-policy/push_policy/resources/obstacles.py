import os
import gym
from glob import glob
import pybullet as p


class Obstacles:
    def __init__(self, client):
        obs_files = glob(os.path.join(os.path.dirname(__file__), 'obstacle*.urdf'))
        self.np_random, _ = gym.utils.seeding.np_random()

        for i in range(4):
            for obs in obs_files:
                x = self.np_random.uniform(-2, 2)
                y = self.np_random.uniform(-2, 2)

                p.loadURDF(fileName=obs,
                           basePosition=[x+0.5, y+0.5, 0],
                           physicsClientId=client)
