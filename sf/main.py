import os
import sys
import argparse
from typing import Optional

from sample_factory.enjoy import enjoy
from sample_factory.train import run_rl
from sample_factory.envs.env_utils import register_env
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args

# making the packages below visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.vizdoomenv import VizDoomGym
from wrappers.render_wrapper import RenderWrapper
from wrappers.glaucoma import GlaucomaWrapper
from wrappers.intrinsic_reward import IntrinsicRewardWrapper
from wrappers.image_transformation import ImageTransformationWrapper

def make_custom_env(full_env_name: str, cfg=None, env_config=None, render_mode: Optional[str] = None):
    env = VizDoomGym(render_mode=render_mode)
    env = ImageTransformationWrapper(env, (161, 161))
    env = GlaucomaWrapper(env, 0, 250, -100)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # env = IntrinsicRewardWrapper(env, RND)
    
    if render_mode == "rgb_array":
        env = RenderWrapper(env)
        # env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    return env

def register_custom_env_envs():
    # register the env in sample-factory's global env registry
    # after this, you can use the env in the command line using --env=custom_env_name
    register_env("health_gathering_glaucoma", make_custom_env)

def add_custom_env_args(_env, p: argparse.ArgumentParser, evaluation=False):
    # You can extend the command line arguments here
    # p.add_argument("--custom_argument", default="value", type=str, help="")
    # p.add_argument(
    #     "--action",
    #     default="play",
    #     choices=["train", "play"],
    #     type=str,
    #     help=(f'choices=["train", "play"]')
    # )
    pass

def custom_env_override_defaults(_env, parser):
    # Modify the default arguments when using this env.
    # These can still be changed from the command line. See configuration guide for more details.
    parser.set_defaults(
        obs_scale=255.0,
        gamma=0.99,
        learning_rate=1e-4,
        lr_schedule="constant",
        adam_eps=1e-5,  
        train_for_env_steps=20_000_000,
        algo="APPO",
        env_frameskip=4,
        use_rnn=True,
        batch_size=2048, 
        num_workers=4, 
        num_envs_per_worker=4, 
        device="cpu",
        num_policies=1,
        experiment="glaucoma250",
        save_video = True,
        video_frames=6000,
    )

def parse_args(argv=None, evaluation=False):
    # parse the command line arguments to build
    parser, partial_cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_custom_env_args(partial_cfg.env, parser, evaluation=evaluation)
    custom_env_override_defaults(partial_cfg.env, parser)
    final_cfg = parse_full_cfg(parser, argv)
    return final_cfg

def train():
    register_custom_env_envs()
    cfg = parse_args()

    status = run_rl(cfg)
    return status

def play():
    register_custom_env_envs()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status

def main():
    """Script entry point."""
    # train()
    play()
    
if __name__ == "__main__":
    sys.exit(main())
