




import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import streamlit as st
import cv2


# Classes pour le modèle (doit correspondre à l'entraînement)
CLASSES = ['Healthy', 'Armillaria_Stage_1', 'Armillaria_Stage_2', 'Armillaria_Stage_3']
DESIRED_SIZE = (224, 224)

class ResNetMultimodal(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ResNetMultimodal, self).__init__()
        
        try:
            self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        except:
            self.resnet = models.resnet18(pretrained=True)
        
        if in_channels != 3:
            original_weight = self.resnet.conv1.weight.data.clone()
            self.resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            with torch.no_grad():
                self.resnet.conv1.weight.data[:, :3] = original_weight
                if in_channels > 3:
                    nn.init.normal_(self.resnet.conv1.weight.data[:, 3:], 
                                  mean=0.0, 
                                  std=original_weight.std().item())
        
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes))
        
    def forward(self, x):
        return self.resnet(x)

@st.cache_resource
def load_resnet_multimodal_model(checkpoint_path="models/multimodal/best_model_resnet.pth"):
    """
    Charge le modèle ResNet multimodal pour la fusion précoce
    """
    try:
        if not os.path.exists(checkpoint_path):
            st.warning(f"Le fichier modèle '{checkpoint_path}' n'existe pas. Utilisation du mode simulation.")
            return None
            
        # Charger le checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Détecter le nombre de canaux d'entrée et de classes
        if 'model_state_dict' in checkpoint:
            # Format avec model_state_dict
            state_dict = checkpoint['model_state_dict']
            in_channels = state_dict['resnet.conv1.weight'].shape[1]
            num_classes = state_dict['resnet.fc.1.weight'].shape[0]
        else:
            # Format avec le modèle complet
            state_dict = checkpoint
            # Essayer de détecter les dimensions à partir des clés du state_dict
            for key in state_dict.keys():
                if 'resnet.conv1.weight' in key:
                    in_channels = state_dict[key].shape[1]
                    break
                elif 'conv1.weight' in key:
                    in_channels = state_dict[key].shape[1]
                    break
            
            # Détecter le nombre de classes
            for key in state_dict.keys():
                if 'resnet.fc.1.weight' in key:
                    num_classes = state_dict[key].shape[0]
                    break
                elif key.endswith('.weight') and len(state_dict[key].shape) == 2:
                    # Supposer que c'est la dernière couche linéaire
                    num_classes = state_dict[key].shape[0]
        
        # Si nous n'avons pas pu détecter, utiliser des valeurs par défaut
        if 'in_channels' not in locals() or in_channels is None:
            in_channels = 8  # RGB (3) + 5 canaux multispectraux
            st.warning(f"Impossible de détecter le nombre de canaux, utilisation par défaut: {in_channels}")
        
        if 'num_classes' not in locals() or num_classes is None:
            num_classes = len(CLASSES)
            st.warning(f"Impossible de détecter le nombre de classes, utilisation par défaut: {num_classes}")
        
        # Créer le modèle
        model = ResNetMultimodal(in_channels=in_channels, num_classes=num_classes)
        
        # Charger les poids
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Mettre en mode évaluation
        model.eval()
        
        st.info(f"ResNet chargé avec {in_channels} canaux d'entrée et {num_classes} classes")
        return model
        
    except Exception as e:
        import traceback
        st.error(f"Erreur lors du chargement du modèle ResNet: {e}")
        st.error(f"Détails: {traceback.format_exc()}")
        return None

def preprocess_early_fusion(rgb_image, ms_images, ms_channels=5):
    """
    Prétraite une image RGB et un ensemble d'images multispectrales pour l'entrée du modèle ResNet
    
    Args:
        rgb_image: Image RGB au format PIL
        ms_images: Liste d'images multispectrales au format PIL
        ms_channels: Nombre de canaux multispectraux à utiliser
    
    Returns:
        Tensor PyTorch prêt pour l'inférence
    """
    # Convertir l'image RGB en tableau numpy et redimensionner
    rgb_img = np.array(rgb_image)
    rgb_img = cv2.resize(rgb_img, DESIRED_SIZE).astype(np.float32) / 255.0
    
    # Vérifier que l'image est bien RGB
    if len(rgb_img.shape) == 2:
        # Convertir grayscale en RGB
        rgb_img = np.stack([rgb_img, rgb_img, rgb_img], axis=2)
    elif rgb_img.shape[2] > 3:
        # Garder seulement les 3 premiers canaux
        rgb_img = rgb_img[:, :, :3]
    
    # Préparer les canaux multispectraux
    ms_stack = []
    for i, img in enumerate(ms_images):
        if i >= ms_channels:
            break
            
        # Convertir en tableau numpy
        img_array = np.array(img)
        
        # Redimensionner
        img_array = cv2.resize(img_array, DESIRED_SIZE)
        
        # Convertir en niveaux de gris si nécessaire
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
        # Normaliser
        img_array = img_array.astype(np.float32) / 255.0
        
        ms_stack.append(img_array)
    
    # Si nous n'avons pas assez d'images MS, compléter avec des zéros
    while len(ms_stack) < ms_channels:
        ms_stack.append(np.zeros(DESIRED_SIZE, dtype=np.float32))
    
    # Réorganiser les canaux pour PyTorch (C, H, W)
    rgb_channels = rgb_img.transpose(2, 0, 1)
    ms_channels_array = np.stack(ms_stack, axis=0)
    
    # Fusionner RGB et MS
    fused = np.concatenate([rgb_channels, ms_channels_array], axis=0)
    
    # Convertir en tensor PyTorch et ajouter la dimension batch
    input_tensor = torch.from_numpy(fused).float().unsqueeze(0)
    
    st.info(f"Tensor préparé pour ResNet: {input_tensor.shape}")
    return input_tensor

def predict_with_resnet(model, input_tensor):
    """
    Fait une prédiction avec le modèle ResNet
    
    Args:
        model: Modèle ResNet chargé
        input_tensor: Tensor d'entrée prétraité
    
    Returns:
        classe prédite et probabilités
    """
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    # Convertir en liste de probabilités pour toutes les classes
    probabilities = probs.squeeze().numpy()
    
    # Retourner la classe prédite et les probabilités
    return CLASSES[pred_class], {cls: float(prob) for cls, prob in zip(CLASSES, probabilities)}