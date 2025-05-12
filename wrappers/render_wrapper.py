import gymnasium as gym
import numpy as np

class RenderWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env):
        self.env = env
        super(RenderWrapper, self).__init__(env)

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset()
        self.last_observation = observation
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_observation = observation 
        return observation, reward, terminated, truncated, info

    def render(self):
        to_render = np.moveaxis(self.last_observation, 0, -1)
        if to_render.shape[-1] == 1:
            to_render = np.repeat(to_render, 3, axis=2)
        return to_render
