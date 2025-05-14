# load libraries
from stable_baselines3.common.env_util import make_vec_env
import torch as th

import gymnasium as gym
from gymnasium.wrappers import RecordVideo

from rllte.xplore.reward import ICM
from rllte.env import make_rllte_env
from rllte.agent import PPO

from env.vizdoomenv import VizDoomGym
from wrappers.render_wrapper import RenderWrapper
from wrappers.glaucoma import GlaucomaWrapper
from wrappers.intrinsic_reward import IntrinsicRewardWrapper
from wrappers.image_transformation import ImageTransformationWrapper

def make_env(render_mode=None):
    env = VizDoomGym(render_mode=render_mode)
    env = ImageTransformationWrapper(env, (161, 161))
    env = GlaucomaWrapper(env, 0, 150, -100)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # env = IntrinsicRewardWrapper(env, RND)
    
    if render_mode == "rgb_array":
        env = RenderWrapper(env)
        env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    return env

# create the vectorized environments
print(isinstance(make_env(), gym.Env))
device = 'cuda' if th.cuda.is_available() else 'cpu'
envs = make_rllte_env(make_env, num_envs=1)
# envs = make_vec_env(make_env, n_envs=1)
print(device, envs.observation_space, envs.action_space)
# create the intrinsic reward module
irs = ICM(envs, device=device)
# create the PPO agent
agent = PPO(envs, device=device)
# set the intrinsic reward module
agent.set(reward=irs)
# train the agent
agent.train(10000)
