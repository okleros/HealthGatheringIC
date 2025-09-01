import os
import sys
import torch

# Make local packages visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.vizdoomenv import VizDoomGym
from wrappers.image_transformation import ImageTransformationWrapper

from sample_factory.algorithms.appo.runner import Runner
from sample_factory.algorithms.appo.config import parse_args
from sample_factory.algorithms.utils.arguments import parse_cfg_args

CHECKPOINT_DIR = "./train/health_gathering"
MODEL_NAME = CHECKPOINT_DIR + "/best_model.pt"

def make_env():
    env = VizDoomGym(render_mode=None)
    env = ImageTransformationWrapper(env, (161, 161))
    return env

def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Sample Factory config
    cfg, _ = parse_cfg_args()
    cfg.env = "vizdoom_custom_env"
    cfg.num_envs = 8  # Parallel environments
    cfg.use_gpu = torch.cuda.is_available()
    cfg.seed = 42
    cfg.total_steps = 2_000_000
    cfg.save_model = True
    cfg.checkpoint_dir = CHECKPOINT_DIR
    cfg.checkpoint_every = 5000
    cfg.resume_training = os.path.exists(MODEL_NAME)

    # Register a custom environment
    from sample_factory.algorithms.appo.envs import register_custom_env

    class CustomVizDoomEnv:
        def __init__(self, **kwargs):
            self.env = make_env()
        # Sample Factory expects reset and step methods
        def reset(self):
            obs, _ = self.env.reset()
            return obs
        def step(self, action):
            obs, reward, done, truncated, info = self.env.step(action)
            return obs, reward, done or truncated, info

    register_custom_env("vizdoom_custom_env", lambda **kwargs: CustomVizDoomEnv(**kwargs))

    # Run training
    args = parse_args()
    runner = Runner(cfg, args)
    runner.run()

if __name__ == "__main__":
    train()
