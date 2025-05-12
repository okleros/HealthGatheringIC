import cv2
import gymnasium as gym
from gymnasium.spaces import Sequence
import numpy as np

class ImageTransformationWrapper(gym.ObservationWrapper):
    def __init__(self, env, resized_shape:Sequence):
        super(ImageTransformationWrapper, self).__init__(env)
        self.resized_shape = resized_shape
        old_shape = self.observation_space.shape
        shape_with_channels = (1, resized_shape[0], resized_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape_with_channels, dtype=np.uint8)

    def observation(self, observation):
        return self.grayscale(observation)
        
    def grayscale(self, observation):
        gray = cv2.cvtColor(np.moveaxis(observation, 0, -1), cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (self.resized_shape[1], self.resized_shape[0]), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (1, self.resized_shape[0], self.resized_shape[1]))
        return state
