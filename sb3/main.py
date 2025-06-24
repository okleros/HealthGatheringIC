import os
import sys
import time
import torch

from gymnasium.wrappers import RecordVideo
from gymnasium.utils.play import play as playing

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from rllte.xplore.reward import RND, E3B

# making the packages below visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.vizdoomenv import VizDoomGym
from wrappers.render_wrapper import RenderWrapper
from wrappers.glaucoma import GlaucomaWrapper
from wrappers.intrinsic_reward import IntrinsicRewardWrapper
from wrappers.image_transformation import ImageTransformationWrapper

CHECKPOINT_DIR = "./train/health_gathering"
LOG_DIR = "./logs/log_health_gathering"
# MODEL_NAME = f"./train/health_gathering_2_2048_grayscale_161x161/best_model_1370000"
# MODEL_NAME = f"./train/health_gathering_3_4096_grayscale_161x161/best_model_970000"
# MODEL_NAME = f"./train/health_gathering_4_4096_grayscale_101x101/best_model_820000"
# MODEL_NAME = f"./train/health_gathering_6_4096_grayscale_161x161_glaucoma50/best_model_1220000"
# MODEL_NAME = f"./train/health_gathering_7_4096_grayscale_161x161_glaucoma150/best_model_390000"
# MODEL_NAME = f"./train/health_gathering_8_4096_grayscale_161x161_glaucoma100/best_model_300000"
# MODEL_NAME = f"./train/health_gathering_9_4096_grayscale_161x161_glaucoma200/best_model_580000"
# MODEL_NAME = f"./train/health_gathering_10_4096_grayscale_161x161_glaucoma250/best_model_1030000"
# MODEL_NAME = f"./train/health_gathering_11_4096_grayscale_161x161_glaucoma250_curiosity/best_model_430000"
# MODEL_NAME = f"./train/health_gathering_12_4096_grayscale_161x161_glaucoma50_curiosity/best_model_880000"
MODEL_NAME = CHECKPOINT_DIR + "/best_model_90000"

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
        # print(observations.shape)
        # print(reward.shape)
        #
        # # ===================== watch the interaction ===================== #
        # self.irs.watch(observations, actions, rewards, dones, dones, next_observations)
        # # ===================== watch the interaction ===================== #
        
        return True
    
    def _on_rollout_end(self) -> None:
        # # ===================== compute the intrinsic rewards ===================== #
        # # prepare the data samples
        obs = torch.as_tensor(self.buffer.observations)
        # # get the new observations
        new_obs = obs.clone()
        new_obs[:-1] = obs[1:]
        new_obs[-1] = torch.as_tensor(self.locals["new_obs"])
        actions = torch.as_tensor(self.buffer.actions)
        rewards = torch.as_tensor(self.buffer.rewards)
        dones = torch.as_tensor(self.buffer.episode_starts)
        # print(obs.dtype)
        # print(new_obs.dtype)
        # print(actions.dtype)
        # print(rewards.dtype)
        # print(dones.dtype)
        # # compute the intrinsic rewards
        # intrinsic_rewards = self.irs.compute(
        #     samples=dict(observations=obs, actions=actions, 
        #                  rewards=rewards, terminateds=dones, 
        #                  truncateds=dones, next_observations=new_obs),
        #     sync=True)
        # # add the intrinsic rewards to the buffer
        # self.buffer.advantages += intrinsic_rewards.cpu().numpy()
        # self.buffer.returns += intrinsic_rewards.cpu().numpy()
        # self.buffer.rewards = np.zeros_like(self.buffer.rewards)
        # self.buffer.rewards += intrinsic_rewards.cpu().numpy()
        # # ===================== compute the intrinsic rewards ===================== #
        pass

def make_env(render_mode=None):
    env = VizDoomGym(render_mode=render_mode)
    env = ImageTransformationWrapper(env, (161, 161))
    # env = GlaucomaWrapper(env, 0, 150, -100)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # env = IntrinsicRewardWrapper(env, RND)
    
    if render_mode == "rgb_array":
        env = RenderWrapper(env)
        env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    return env

def play():
    env = make_env(render_mode="human")
    model = PPO.load(MODEL_NAME)
    # model.summary();
    
    episodes = 5
    for episode in range(episodes):
        total_reward = 0
        finished = False
        obs, _ = env.reset()
        while not finished:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            # time.sleep(0.001)
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

    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=50)

    env.close()

    print(mean_reward)

def train():
    print("Training")
    envs = make_vec_env(make_env, n_envs=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    irs = RND(envs, device=device)
    callback = TrainAndLoggingCallback(irs=irs, check_freq=10000, save_path=CHECKPOINT_DIR)
    
    model = PPO("CnnPolicy", envs, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=4096)
    
    model.learn(total_timesteps=2_000_000, callback=callback, progress_bar=True)
    
    envs.close()
    

def callback_playing(obs_t, obs_tp1, action, reward, terminated, truncated, info):
    # cv2.imwrite("output.png", np.moveaxis(obs_tp1, 0, -1))
    print(reward)
    # print(info)

def play_human():
    env = make_env(render_mode="rgb_array")
    playing(env, keys_to_action={ "a": 0, "d": 1, "w": 2 }, wait_on_player=True, callback=callback_playing)
    env.close()


if __name__ == "__main__":
    # train()
    # evaluate()
    # play()
    # record()
    # play_human()

    print(MODEL_NAME)
    evaluate()
    for i in range(100000, 170000, 10000):
        MODEL_NAME = MODEL_NAME.replace(str(i-10000), str(i))
        print(MODEL_NAME)
        # print("mean_reward model " + str(i) + ":")
        evaluate()
