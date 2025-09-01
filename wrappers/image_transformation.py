# import cv2
# import gymnasium as gym
# from gymnasium.spaces import Sequence
# import numpy as np

# class ImageTransformationWrapper(gym.ObservationWrapper):
#     def __init__(self, env, resized_shape:Sequence):
#         super(ImageTransformationWrapper, self).__init__(env)
#         self.resized_shape = resized_shape
#         old_shape = self.observation_space.shape
#         shape_with_channels = (1, resized_shape[0], resized_shape[1])
#         self.observation_space = gym.spaces.Box(
#             low=0, high=255, shape=shape_with_channels, dtype=np.uint8)

#     def observation(self, observation):
#         return self.grayscale(observation)
        
#     def grayscale(self, observation):
#         gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
#         resize = cv2.resize(gray, (self.resized_shape[1], self.resized_shape[0]), interpolation=cv2.INTER_AREA)
#         state = np.reshape(resize, (1, self.resized_shape[0], self.resized_shape[1]))
#         return state

import cv2
import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box, Sequence

class ImageTransformationWrapper(gym.ObservationWrapper):
    def __init__(self, env, resized_shape: Sequence):
        super().__init__(env)
        self.resized_shape = resized_shape
        c, h, w = 1, resized_shape[0], resized_shape[1]
        self.observation_space = Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

    def observation(self, observation):
        return self.grayscale(observation)

    def grayscale(self, observation):
        # Convert channel-first (C,H,W) to channel-last (H,W,C) for OpenCV
        obs_ch_last = np.moveaxis(observation, 0, -1)
        gray = cv2.cvtColor(obs_ch_last, cv2.COLOR_BGR2GRAY)
        # Resize using fast INTER_AREA interpolation
        resize = cv2.resize(gray, (self.resized_shape[1], self.resized_shape[0]), interpolation=cv2.INTER_AREA)
        # Add channel dimension back (C,H,W)
        return resize[None, :, :]
