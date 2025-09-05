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
MODEL_NAME = CHECKPOINT_DIR + "/best_model_45000"

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
    env = ImageTransformationWrapper(env, (128, 128))
    
    if render_mode == "rgb_array":
        env = RenderWrapper(env)
        env = RecordVideo(env, video_folder="./videos", episode_trigger=lambda x: True)
    return env

def make_env_single_frame():
    env = VizDoomGym()
    env = ImageTransformationWrapper(env, (128, 128))
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

    original_height, original_width = 128, 128
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
    
    # Define a mapping from action index to a human-readable action name.
    # Adjust this based on your specific VizDoomGym action space.
    action_meanings = {
        0: "LEFT", 1: "RIGHT", 2: "FORWARD",
        # Add other actions as necessary if your action space is larger
        # e.g., 3: "BACKWARD", 4: "TURN_LEFT", 5: "TURN_RIGHT", etc.
    }
    
    for i in range(num_frames_to_process):
        # 1. Preparar a observação para o modelo
        input_tensor = torch.as_tensor(obs_frame).float().unsqueeze(0).to(model.device)

        # --- Get Network Outputs (Logits and Probabilities) ---
        with torch.no_grad():
            features_full_pred = model.policy.features_extractor(input_tensor)
            action_logits = model.policy.action_net(features_full_pred) # Raw scores
            action_probs = F.softmax(action_logits, dim=1) # Probabilities
            
            predicted_action_index = torch.argmax(action_logits, dim=1).item()
            predicted_action_name = action_meanings.get(predicted_action_index, f"Action {predicted_action_index}")

        # 2. Define the target_category for Grad-CAM (e.g., class 2)
        # Using the predicted action as the target category for Grad-CAM
        # target_category_for_gradcam = predicted_action_index
        # OR keep fixed to class 2 as previously requested
        target_category_for_gradcam = None
        
        # 3. Calculate the heatmap for the specified target_category
        heatmap_raw = grad_cam_instance(input_tensor, target_category=target_category_for_gradcam) 
        
        # 4. Post-process heatmap for binarization (mask generation)
        heatmap_resized = cv2.resize(heatmap_raw, new_resolution_tuple, interpolation=cv2.INTER_LINEAR)
        heatmap_8bit = np.uint8(255 * heatmap_resized)
        
        threshold_value = 128 
        ret, gradcam_mask = 0, heatmap_8bit
        
        # --- Prepare Original Frame in BGR and Resized ---
        img_original_bgr = np.moveaxis(obs_frame, 0, -1)
        img_original_bgr = cv2.cvtColor(img_original_bgr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        img_original_resized = cv2.resize(img_original_bgr, new_resolution_tuple, interpolation=cv2.INTER_NEAREST)
        
        # --- Multiplicação do frame original colorida pela MÁSCARA binarizada ---
        gradcam_mask_3_channels = cv2.cvtColor(gradcam_mask, cv2.COLOR_GRAY2BGR) / 255.0
        highlighted_areas_image = (img_original_resized*gradcam_mask_3_channels).astype(np.uint8)

        # --- Annotate Network Outputs on Highlighted Frame ---
        # Create a copy of the original image to annotate without modifying the original for concatenation
        annotated_original_image = img_original_resized.copy()
        
        # Add annotations to the original frame
        text_annotation_y_start = 30 # Starting Y position for annotations
        text_line_height = 25 # Height for each line of text
        
        # Annotate Predicted Action on ORIGINAL FRAME
        text_pred_action_orig = f"Predicted Action: {predicted_action_name}"
        cv2.putText(annotated_original_image, text_pred_action_orig, (10, text_annotation_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA) # White
        text_annotation_y_start += text_line_height

        # Annotate Probabilities for ALL Actions on ORIGINAL FRAME
        for action_idx in range(action_probs.shape[1]):
            action_name = action_meanings.get(action_idx, f"Action {action_idx}")
            prob = action_probs[0, action_idx].item()
            
            color = (0, 255, 0) # Green for regular probabilities
            if action_idx == predicted_action_index:
                color = (0, 0, 255) # Red for the predicted action
            elif action_idx == target_category_for_gradcam:
                # Use a different color if Grad-CAM target is not the predicted action
                if action_idx != predicted_action_index: 
                    color = (255, 255, 0) # Cyan for Grad-CAM target class
                else: # If target is also predicted, keep it red.
                    color = (0, 0, 255)
            
            text_prob = f"{action_name}: {prob:.3f}"
            cv2.putText(annotated_original_image, text_prob, (10, text_annotation_y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
            y_pos_for_merge = text_annotation_y_start
            text_annotation_y_start += text_line_height

        # Annotate Grad-CAM target category on HIGHLIGHTED FRAME
        # text_gradcam_target = f"Grad-CAM Target: {action_meanings.get(target_category_for_gradcam, f'Action {target_category_for_gradcam}')}"
        # cv2.putText(highlighted_areas_image, text_gradcam_target, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA) # Cyan

        # --- Join Images Horizontally and Save ---
        # Ensure both images are the same height. They should be due to previous resizing.
        # If not, you might need to resize them here again.
        if annotated_original_image.shape[0] != highlighted_areas_image.shape[0]:
            print("Warning: Image heights differ. Resizing for concatenation.")
            h = max(annotated_original_image.shape[0], highlighted_areas_image.shape[0])
            annotated_original_image = cv2.resize(annotated_original_image, (int(annotated_original_image.shape[1] * h / annotated_original_image.shape[0]), h))
            highlighted_areas_image = cv2.resize(highlighted_areas_image, (int(highlighted_areas_image.shape[1] * h / highlighted_areas_image.shape[0]), h))

        # Join horizontally
        merged_image_output = cv2.hconcat([annotated_original_image, highlighted_areas_image])
        
        # Save the merged image
        filename_merged = os.path.join(output_dir, f'merged_output_frame_{i:03d}.png')
        cv2.imwrite(filename_merged, merged_image_output)

        # 6. Step the environment
        action_env, _ = model.predict(obs_frame, deterministic=True)
        obs_frame, reward, done, truncated, info = env_test.step(action_env)
        
        if done or truncated:
            print(f"Episódio terminado no frame {i}. Resetando ambiente.")
            obs_frame, _ = env_test.reset()

    print(f"Processamento de {num_frames_to_process} frames concluído.")
    env_test.close()