import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --- Definição das Classes GradCAM e PolicyHeadForGradCAM (se ainda não estiverem definidas) ---

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.model.eval() # Coloca o modelo em modo de avaliação
        self.activations = None
        self.gradients = None

        # Registra hooks para capturar ativações e gradientes
        self.target_layer.register_forward_hook(self._save_activations)
        self.target_layer.register_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0] # grad_output é uma tupla, pegamos o primeiro elemento

    def __call__(self, input_tensor, target_category=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_category is None:
            target_category = torch.argmax(output).item()
        
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_category] = 1
        output.backward(gradient=one_hot_output, retain_graph=True)

        if len(self.gradients.shape) == 4:
            pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        elif len(self.gradients.shape) == 2:
            pooled_gradients = torch.mean(self.gradients, dim=0)
        else:
            raise ValueError(f"Unexpected gradients shape for Grad-CAM: {self.gradients.shape}")

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

# --- Lógica Principal para aplicar Grad-CAM em um Frame e Plotar Bounding Boxes ---

def visualize_gradcam_on_frame(model, obs_frame, target_layer, output_dim, render_size=(161, 161), threshold_value=128, min_contour_area=200):
    """
    Aplica Grad-CAM em um único frame, gera o heatmap e plota a imagem original
    com o heatmap e as bounding boxes.

    Args:
        model: O modelo PPO carregado.
        obs_frame (np.array): Um único frame de observação (C, H, W).
        target_layer (torch.nn.Module): A última camada convolucional para Grad-CAM.
        output_dim (int): A dimensão do output da rede (número de ações/classes).
        render_size (tuple): Tamanho (largura, altura) para redimensionar o heatmap e a imagem.
        threshold_value (int): Valor de limiar para binarizar o heatmap (0-255).
        min_contour_area (int): Área mínima do contorno para desenhar a bounding box.
    """
    # 1. Preparar a observação para o modelo
    input_tensor = torch.as_tensor(obs_frame).float().unsqueeze(0).to(model.device)

    # 2. Instanciar PolicyHeadForGradCAM e GradCAM
    policy_head_for_gradcam = PolicyHeadForGradCAM(
        model.policy.features_extractor,
        model.policy.action_net
    ).to(model.device)

    grad_cam_instance = GradCAM(policy_head_for_gradcam, target_layer)

    # 3. Obter a predição da categoria alvo
    with torch.no_grad():
        features_full = model.policy.features_extractor(input_tensor)
        action_logits_pred = model.policy.action_net(features_full)
        predicted_category = torch.argmax(action_logits_pred, dim=1).item()

    # 4. Calcular o heatmap
    heatmap = grad_cam_instance(input_tensor, target_category=predicted_category)

    # 5. Pós-processamento do heatmap e obtenção das bounding boxes
    # Converter a observação original para o formato (H, W, C) e BGR para OpenCV
    img_bgr = np.moveaxis(obs_frame, 0, -1) # De (C, H, W) para (H, W, C)
    img_bgr = cv2.cvtColor(img_bgr.astype(np.uint8), cv2.COLOR_RGB2BGR) # Converte para BGR

    # Redimensionar heatmap para o tamanho da observação original
    h, w, _ = img_bgr.shape
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_8bit = np.uint8(255 * heatmap_resized) # Mapa de calor de 1 canal, 0-255

    # Aplicar limiarização
    ret, thresh = cv2.threshold(heatmap_8bit, threshold_value, 255, cv2.THRESH_BINARY)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copiar a imagem original para desenhar as boxes
    img_with_boxes = img_bgr.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > min_contour_area:
            x, y, box_w, box_h = cv2.boundingRect(cnt)
            cv2.rectangle(img_with_boxes, (x, y), (x + box_w, y + box_h), (0, 255, 0), 2) # Verde

    # Opcional: Sobrepor o heatmap colorido na imagem original para visualização
    heatmap_colored = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET) # Heatmap colorido
    # Misturar a imagem original com o heatmap colorido
    # Primeiro, converta a imagem original para o mesmo tipo e normalização do heatmap_colored (0-255, uint8)
    # E para RGB, já que Matplotlib espera RGB
    superimposed_img = cv2.addWeighted(img_bgr, 0.6, heatmap_colored, 0.4, 0)
    superimposed_img_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)


    # 6. Plotar os resultados
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.title('Frame Original')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_resized, cmap='jet') # Heatmap em escala de cinza com colormap
    plt.title('Mapa de Calor Grad-CAM')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)) # Imagem com boxes
    plt.title('Frame com Bounding Boxes')
    plt.axis('off')
    
    plt.figure(figsize=(8, 8))
    plt.imshow(superimposed_img_rgb)
    plt.title('Frame com Heatmap Sobreposto')
    plt.axis('off')

    plt.show()

# --- Exemplo de como usar a função visualize_gradcam_on_frame ---
if __name__ == "__main__":
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    import os
    import sys

    # Ajuste o sys.path para encontrar seus wrappers e env
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from env.vizdoomenv import VizDoomGym
    from wrappers.image_transformation import ImageTransformationWrapper

    # Definições do seu ambiente e modelo
    MODEL_NAME = "./train/health_gathering/best_model_90000" # Use seu caminho correto
    
    def make_env_single(render_mode=None): # Função para criar um único ambiente
        env = VizDoomGym(render_mode=render_mode)
        env = ImageTransformationWrapper(env, (161, 161))
        return env

    # Carregar o modelo
    env_test_single = make_env_single()
    model = PPO.load(MODEL_NAME, env=env_test_single)
    
    # Identificar a última camada convolucional
    # model.policy.features_extractor é o extrator de features completo
    # Sua parte CNN está em model.policy.features_extractor.cnn
    cnn_part = model.policy.features_extractor.cnn 
    
    target_layer = None
    for name, module in cnn_part.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module # A última camada Conv2d
    if target_layer is None:
        raise ValueError("No convolutional layer found in the CNN feature extractor.")

    # Obter um frame de observação de exemplo
    obs, _ = env_test_single.reset()
    
    # Definir output_dim (número de ações)
    output_dim = env_test_single.action_space.n

    # Chamar a função de visualização
    visualize_gradcam_on_frame(model, obs, target_layer, output_dim, 
                                threshold_value=150, min_contour_area=300) # Ajuste esses valores!
    
    env_test_single.close()