import os
import sys
import time
import torch

import torch.nn.functional as F
import cv2
import numpy as np

from gymnasium.wrappers import RecordVideo
from gymnasium.utils.play import play as playing

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

from rllte.xplore.reward import RND, E3B

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from env.vizdoomenv import VizDoomGym
from wrappers.render_wrapper import RenderWrapper
from wrappers.glaucoma import GlaucomaWrapper
from wrappers.intrinsic_reward import IntrinsicRewardWrapper
from wrappers.image_transformation import ImageTransformationWrapper

CHECKPOINT_DIR = "./train/health_gathering"
LOG_DIR = "./logs/log_health_gathering"
MODEL_NAME = CHECKPOINT_DIR + "/best_model_100000"

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval()
        self.activations = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_category is None:
            target_category = torch.argmax(output).item()
        
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_category] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        weighted_activations = self.activations.clone()
        for i in range(weighted_activations.size(1)):
            weighted_activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(weighted_activations, dim=1).squeeze()
        heatmap = F.relu(heatmap)
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        else:
            heatmap = torch.zeros_like(heatmap)

        return heatmap.detach().cpu().numpy()

class PolicyHeadForGradCAM(torch.nn.Module):
    def __init__(self, full_feature_extractor, action_net):
        super().__init__()
        self.full_feature_extractor = full_feature_extractor
        self.action_net = action_net
    
    def forward(self, x):
        features = self.full_feature_extractor(x)
        action_logits = self.action_net(features)
        return action_logits

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
        return True
    
    def _on_rollout_end(self) -> None:
        pass

def make_env(render_mode=None):
    env = VizDoomGym(render_mode=render_mode)
    env = ImageTransformationWrapper(env, (161, 161))
    
    if render_mode == "rgb_array":
        env = RenderWrapper(env)
        env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    return env

def make_env_single_frame():
    env = VizDoomGym()
    env = ImageTransformationWrapper(env, (161, 161))
    return env

def play():
    env = make_env(render_mode="human")
    model = PPO.load(MODEL_NAME)
    
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
    envs = make_vec_env(make_env, n_envs=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    irs = RND(envs, device=device)
    callback = TrainAndLoggingCallback(irs=irs, check_freq=10000, save_path=CHECKPOINT_DIR)
    
    model = PPO("CnnPolicy", envs, tensorboard_log=LOG_DIR, learning_rate=0.0001, n_steps=4096)
    
    model.learn(total_timesteps=2_000_000, callback=callback, progress_bar=True)
    
    envs.close()
    

def callback_playing(obs_t, obs_tp1, action, reward, terminated, truncated, info):
    print(reward)

def play_human():
    env = make_env(render_mode="rgb_array")
    playing(env, keys_to_action={ "a": 0, "d": 1, "w": 2 }, wait_on_player=True, callback=callback_playing)
    env.close()


# --- Bloco de Execução Principal para Grad-CAM em 100 frames ---
if __name__ == "__main__":
    print(f"Carregando o modelo PPO de: {MODEL_NAME}")
    env_test = make_env_single_frame()
    model = PPO.load(MODEL_NAME, env=env_test)
    print("Modelo carregado com sucesso.")

    cnn_part_for_gradcam_target = model.policy.features_extractor.cnn 
    print("\n--- Arquitetura da Rede CNN (features_extractor.cnn) ---")
    print(cnn_part_for_gradcam_target)
    print("-------------------------------------------------------\n")

    target_layer = None
    for name, module in cnn_part_for_gradcam_target.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module
    if target_layer is None:
        raise ValueError("Nenhuma camada convolucional encontrada no extrator de features da CNN.")
    print(f"Última camada convolucional identificada: {target_layer}")

    output_dim = env_test.action_space.n 

    output_dir = "gradcam_output_highlighted_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Imagens de frames destacados serão salvas em: {os.path.abspath(output_dir)}")

    original_height, original_width = 161, 161
    new_resolution_width = original_width * 4
    new_resolution_height = original_height * 4
    new_resolution_tuple = (new_resolution_width, new_resolution_height)

    policy_head_for_gradcam = PolicyHeadForGradCAM(
        model.policy.features_extractor,
        model.policy.action_net
    ).to(model.device)
    grad_cam_instance = GradCAM(policy_head_for_gradcam, target_layer)

    num_frames_to_process = 500
    print(f"Processando {num_frames_to_process} frames e salvando as áreas de interesse destacadas...")
    
    obs_frame, _ = env_test.reset()
    
    for i in range(num_frames_to_process):
        # 1. Preparar a observação para o modelo
        input_tensor = torch.as_tensor(obs_frame).float().unsqueeze(0).to(model.device)

        # 2. Obter a predição da categoria alvo (classe 2)
        target_category = None
        
        # 3. Calcular o heatmap
        heatmap_raw = grad_cam_instance(input_tensor, target_category=target_category) 
        
        # 4. Pós-processamento do heatmap para binarização
        heatmap_resized = cv2.resize(heatmap_raw, new_resolution_tuple, interpolation=cv2.INTER_LINEAR)
        heatmap_8bit = np.uint8(255 * heatmap_resized)
        
        threshold_value = 128 
        ret, gradcam_threshold_image = 0, heatmap_8bit#cv2.threshold(heatmap_8bit, threshold_value, 255, cv2.THRESH_BINARY)
        
        # --- Processar e Salvar o Frame Original ---
        img_original_bgr = np.moveaxis(obs_frame, 0, -1) # De (C, H, W) para (H, W, C)
        img_original_bgr = cv2.cvtColor(img_original_bgr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_original_resized = cv2.resize(img_original_bgr, new_resolution_tuple, interpolation=cv2.INTER_NEAREST)
        
        # filename_original = os.path.join(output_dir, f'original_frame_{i:03d}.png')
        # cv2.imwrite(filename_original, img_original_resized)

        # --- Multiplicação do frame original colorida pelo heatmap binarizado ---
        # Expande o heatmap binarizado (1 canal) para 3 canais para a multiplicação
        gradcam_mask_3_channels = cv2.cvtColor(gradcam_threshold_image, cv2.COLOR_GRAY2BGR)/255.0
        
        # Multiplicação elemento a elemento
        # Isso irá "mascarar" o frame original, deixando visível apenas onde o heatmap é branco.
        highlighted_areas_image = (img_original_resized*gradcam_mask_3_channels).astype(np.uint8)

        # Salvar a imagem com as áreas de interesse destacadas
        filename_highlighted = os.path.join(output_dir, f'highlighted_frame_{i:03d}.png')
        # cv2.imwrite(filename_highlighted, highlighted_areas_image)

        image_horizontal = cv2.hconcat([img_original_resized, highlighted_areas_image])
        cv2.imwrite(filename_highlighted, image_horizontal)

        # 6. Dar um passo no ambiente para obter o próximo frame
        action, _ = model.predict(obs_frame)
        obs_frame, reward, done, truncated, info = env_test.step(action)
        
        if done or truncated:
            print(f"Episódio terminado no frame {i}. Resetando ambiente.")
            obs_frame, _ = env_test.reset()

    print(f"Processamento de {num_frames_to_process} frames concluído.")
    env_test.close()