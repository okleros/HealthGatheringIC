from typing import List
from vizdoom import *

import os
import cv2
import time
import torch
import random
import numpy as np

from collections.abc import Sequence

from rllte.xplore.reward import RND, E3B

import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import RecordVideo
from gymnasium.utils.play import play as playing
from gymnasium.spaces import Discrete, Box, Sequence

from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

CHECKPOINT_DIR = "./train/health_gathering"
LOG_DIR = "./logs/log_health_gathering"
# MODEL_NAME = f"./train/health_gathering_2_2048_grayscale_161x161/best_model_1370000"
# MODEL_NAME = f"./train/health_gathering_3_4096_grayscale_161x161/best_model_970000"
# MODEL_NAME = f"./train/health_gathering_4_4096_grayscale_101x101/best_model_820000"
# MODEL_NAME = f"./train/health_gathering_6_4096_grayscale_161x161_glaucoma50/best_model_1220000"
# MODEL_NAME = f"./train/health_gathering_7_4096_grayscale_161x161_glaucoma150/best_model_390000"
# MODEL_NAME = f"./train/health_gathering_8_4096_grayscale_161x161_glaucoma100/best_model_300000"
# MODEL_NAME = f"./train/health_gathering_9_4096_grayscale_161x161_glaucoma200/best_model_580000"
MODEL_NAME = f"./train/health_gathering_10_4096_grayscale_161x161_glaucoma250/best_model_1030000"


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

class GlaucomaWrapper(gym.Wrapper):
    def __init__(self, env:gym.Env, steps_with_hungry_to_glaucoma:int, steps_glaucoma_level:int, 
                 blidness_reward:float):
        """
        steps_with_hungry_to_glaucoma: how much steps the agent will be with hungry before glaucoma begins
        steps_glaucoma_level: how much pixels the glaucoma will take when the agent is hungry
        blidness_reward: the reward the agent will win after blidness, the vision after blidness will be reseted
        """
        # env
        self.env = env
        super(GlaucomaWrapper, self).__init__(env)

        # steps heuristic
        self.steps_with_hungry_to_glaucoma = steps_with_hungry_to_glaucoma
        self.steps_with_hungry = 0
        self.steps_glaucoma_level = steps_glaucoma_level

        # blidness reward
        self.blindness_reward = blidness_reward
        self.blind = False

        # pixel stuffs
        self.pixels = self.generate_spiral(env.observation_space.shape[1], env.observation_space.shape[2])
        self.erased_pixel = 0

        self.last_medkits = 0

    def reset(self, seed=None, options=None):
        self.blind = False
        self.steps_with_hungry = 0
        self.erased_pixel = 0
        self.last_medkits = 0
        return self.env.reset()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.glaucoma_policy(info)
            
        return self.erase_pixels(observation), reward, terminated, truncated, info
    
    def glaucoma_policy(self, info):
        medkit_used = False 
        if info["medkits_used"] > self.last_medkits:
            medkit_used = True
        self.last_medkits = info["medkits_used"]
        
        if medkit_used:
            self.steps_with_hungry = -1
            self.erased_pixel = 0
        self.steps_with_hungry += 1

        if self.steps_with_hungry > self.steps_with_hungry_to_glaucoma:
            self.erased_pixel += self.steps_glaucoma_level
            if self.erased_pixel > len(self.pixels):
                self.blind = True

    def reward_policy(self, reward):
        if self.blind:
            reward = self.blindness_reward
            self.erased_pixel = 0
        self.blind = False
        return reward

    def erase_pixels(self, observation):
        if self.erased_pixel > 0:
            rows, cols = zip(*self.pixels[:self.erased_pixel])
            observation[:, rows, cols] = 0
        return observation

    def generate_spiral(self, n, m):
        x = n//2
        y = x
        # RIGHT, DOWN, LEFT, UP
        dx = [0, 1, 0, -1]
        dy = [1, 0, -1, 0]
        pixels = [(x, y)]
        flag = True
        count = 1
        count_of_count = 0
        direction = 0
        while flag:
            for c in range(count):
                x += dx[direction]
                y += dy[direction]
                if is_inside(x, y, n, m):
                    pixels.append((x, y))
                else:
                    flag = False      
                    break
            count_of_count =  (count_of_count+1)%2
            count += (count_of_count==0)
            direction = (direction+1)%4
        return pixels

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, irs, check_freq:int, save_path:str, verbose:int = 0):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.irs = irs
        self.check_freq = check_freq
        self.save_path = save_path
        self.buffer = None

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.buffer = self.model.rollout_buffer
            

    def _on_step(self) -> bool:
        if self.n_calls%self.check_freq == 0:
            model_path = os.path.join(self.save_path, f"best_model_{self.n_calls}")
            self.model.save(model_path)

        # observations = self.locals["obs_tensor"]
        # device = observations.device
        # actions = torch.as_tensor(self.locals["actions"], device=device)
        # rewards = torch.as_tensor(self.locals["rewards"], device=device)
        # dones = torch.as_tensor(self.locals["dones"], device=device)
        # next_observations = torch.as_tensor(self.locals["new_obs"], device=device)
        #
        # # ===================== watch the interaction ===================== #
        # self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # # ===================== watch the interaction ===================== #
        
        return True
    
    def _on_rollout_end(self) -> None:
        # # ===================== compute the intrinsic rewards ===================== #
        # # prepare the data samples
        # obs = torch.as_tensor(self.buffer.observations)
        # # get the new observations
        # new_obs = obs.clone()
        # new_obs[:-1] = obs[1:]
        # new_obs[-1] = torch.as_tensor(self.locals["new_obs"])
        # actions = torch.as_tensor(self.buffer.actions)
        # rewards = torch.as_tensor(self.buffer.rewards)
        # dones = torch.as_tensor(self.buffer.episode_starts)
        # # compute the intrinsic rewards
        # intrinsic_rewards = self.irs.compute(
        #     samples=dict(observations=obs, actions=actions, 
        #                  rewards=rewards, terminateds=dones, 
        #                  truncateds=dones, next_observations=new_obs),
        #     sync=True)
        # # add the intrinsic rewards to the buffer
        # self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        # self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # # ===================== compute the intrinsic rewards ===================== #
        # # print(self.buffer.rewards)
        # # print(intrinsic_rewards.cpu().numpy())
        pass

def make_env(render_mode=None):
    env = VizDoomGym(render_mode=render_mode)
    env = ImageTransformationWrapper(env, (161, 161))
    env = GlaucomaWrapper(env, 0, 250, -100)
    if render_mode == "rgb_array":
        env = RenderWrapper(env)
        env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    return env


def play():
    env = make_env(render_mode="human")
    model = PPO.load(MODEL_NAME)
    model.rollout_buffer.rewards
    
    episodes = 5
    for episode in range(episodes):
        total_reward = 0
        finished = False
        obs, _ = env.reset()
        while not finished:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            time.sleep(0.05)
            total_reward += reward
            finished = done or truncated
        print(f"Total Reward for episode {episode} is {total_reward}.")
        time.sleep(2)

    env.close()

def record():
    env = make_env(render_mode="rgb_array")
    model = PPO.load(MODEL_NAME)
    
    episodes = 5
    for episode in range(episodes):
        total_reward = 0
        finished = False
        obs, _ = env.reset()
        while not finished:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            finished = done or truncated
        print(f"Total Reward for episode {episode} is {total_reward}.")

    env.close()
    
def evaluate():
    model = PPO.load(MODEL_NAME)
    
    env = make_env()

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)

    env.close()

    print(mean_reward)

def train():
    print("Training")
    envs = make_vec_env(make_env, n_envs=1)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    irs = RND(envs, device=device)
    callback = TrainAndLoggingCallback(irs=irs, check_freq=10000, save_path=CHECKPOINT_DIR)
    
    model = PPO("CnnPolicy", envs, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=4096)
    
    model.learn(total_timesteps=2000000, callback=callback, progress_bar=True)
    
    envs.close()
    

def callback_playing(obs_t, obs_tp1, action, reward, terminated, truncated, info):
    cv2.imwrite("output.png", np.moveaxis(obs_tp1, 0, -1))
    # print(reward)
    # print(info)

def play_human():
    env = make_env(render_mode="rgb_array")
    playing(env, keys_to_action={ "a": 0, "d": 1, "w": 2 }, wait_on_player=True, callback=callback_playing)
    env.close()

def is_inside(x, y, n, m):
    return x >= 0 and x < n and y >= 0 and y < m

if __name__ == "__main__":
    # train()
    # evaluate()
    # play()
    record()
    # play_human()
