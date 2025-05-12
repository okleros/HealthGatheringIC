import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.env_util import make_vec_env

class IntrinsicRewardWrapper(gym.Wrapper):
    def __init__(self, env, intrinsic_module):
        super().__init__(env)
        vectorized_env = make_vec_env(lambda: env, 1)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.intrinsic_module = intrinsic_module(vectorized_env, self.device)
        self.prev_obs = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_obs = obs
        return obs, info

    def step(self, action):
        # return self.env.step(action)
        obs, extrinsic_reward, terminated, truncated, info = self.env.step(action)

        # Prepare transition and compute intrinsic reward
        transition = {
            "observations": torch.tensor(np.expand_dims(self.prev_obs, axis=0)).unsqueeze(0).to(device=self.device, dtype=torch.float32),
            "next_observations": torch.tensor(np.expand_dims(obs, axis=0)).unsqueeze(0).to(device=self.device, dtype=torch.float32),
            "actions": torch.tensor([action]).unsqueeze(0).to(device=self.device, dtype=torch.float32),
            "rewards": torch.tensor([extrinsic_reward]).unsqueeze(0).to(device=self.device, dtype=torch.float32),
            "terminateds": torch.tensor([terminated]).unsqueeze(0).to(device=self.device, dtype=torch.float32),
            "truncateds": torch.tensor([truncated]).unsqueeze(0).to(device=self.device, dtype=torch.float32),
        }
        print("HEREEEEEEEEEEEEEEEEEEEEEEEEEEE")
        print(transition["observations"].shape)
        print(transition["next_observations"].shape)
        print(transition["actions"].shape)
        print(transition["rewards"].shape)
        print(transition["terminateds"].shape)
        print(transition["truncateds"].shape)
        print(self.intrinsic_module.compute(transition))

        intrinsic_reward = self.intrinsic_module.compute(transition)[0].item()

        # total_reward = extrinsic_reward + intrinsic_reward
        total_reward = intrinsic_reward
        self.prev_obs = obs

        return obs, total_reward, terminated, truncated, info
