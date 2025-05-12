import gymnasium as gym
from vizdoom import *
from gymnasium.spaces import Discrete, Box, Sequence
from gymnasium import Env, make
import numpy as np

class VizDoomGym(Env):
    metadata = { "render_modes": ["human", "rgb_array"], "render_fps": 30 }

    def __init__(self, render_mode=None) -> None:
        super().__init__()
        
        self.game = DoomGame()
        self.game.load_config("./scenarios/health_gathering.cfg")
        
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        if render_mode != "human":
            self.game.set_window_visible(False)
        else:
            self.game.set_window_visible(True)
        self.game.init()

        # set of actions we can take in the enviroment
        self.actions = np.identity(3, dtype=np.uint8)

        # spaces
        self.observation_space = Box(low=0, high=255, shape=(3, 240, 320), dtype=np.uint8)
        self.action_space = Discrete(3)

    def reset(self, seed=None, options=None):
        self.game.new_episode()
        state = self.game.get_state()
        return state.screen_buffer, { "ammo": state.game_variables[0], "medkits_used": 0 }
    
    def step(self, action):
        # doing step
        reward = self.game.make_action(self.actions[action], 4)
        
        # getting the other information
        state = self.game.get_state()
        if self.game.get_state():
            img = state.screen_buffer
            info = { "ammo": state.game_variables[0], "medkits_used": int(state.game_variables[1]) }
        else:
            img = np.zeros(self.observation_space.shape, dtype=np.uint8)
            info = { "ammo": 0, "medkits_used": 0 }

        done = self.game.is_episode_finished()
        
        return img, reward, done, False, info

    def render(self):
        state = self.game.get_state()
        if state:
            img = state.screen_buffer
        else:
            img = np.zeros(self.observation_space.shape, dtype=np.uint8)
        return np.moveaxis(img, 0, -1)

    def close(self):
        self.game.close()
