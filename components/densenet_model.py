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

class DenseNetMultimodal(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(DenseNetMultimodal, self).__init__()
        densenet = models.densenet121(pretrained=True)
        original_conv = densenet.features.conv0
        
        # Création d'une nouvelle couche de convolution pour gérer plus de canaux
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None
        )
        
        # Initialisation des poids pour les canaux RGB
        with torch.no_grad():
            new_conv.weight[:, :3] = original_conv.weight
            # Initialisation pour les canaux multispectraux supplémentaires
            if in_channels > 3:
                for i in range(3, in_channels):
                    #new_conv.weight[:, i] = original_conv.weight[:, :3].mean(dim=1, keepdim=True)
                    new_conv.weight[:, i] = original_conv.weight[:, :3].mean(dim=1)

        
        densenet.features.conv0 = new_conv
        self.features = densenet.features
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes))
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

@st.cache_resource
def load_densenet_multimodal_model(checkpoint_path="models/multimodal/best_model_densenet.pth"):
    """
    Charge le modèle DenseNet multimodal pour la fusion précoce
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
            in_channels = state_dict['features.conv0.weight'].shape[1]
            num_classes = state_dict['classifier.1.weight'].shape[0]
        else:
            # Format avec le modèle complet
            state_dict = checkpoint
            # Essayer de détecter les dimensions à partir des clés du state_dict
            for key in state_dict.keys():
                if 'features.conv0.weight' in key:
                    in_channels = state_dict[key].shape[1]
                    break
                elif 'conv0.weight' in key:
                    in_channels = state_dict[key].shape[1]
                    break
            
            # Détecter le nombre de classes
            for key in state_dict.keys():
                if 'classifier.1.weight' in key:
                    num_classes = state_dict[key].shape[0]
                    break
                elif key.endswith('.weight') and len(state_dict[key].shape) == 2:
                    # Supposer que c'est la dernière couche linéaire
                    num_classes = state_dict[key].shape[0]
        
        # Si nous n'avons pas pu détecter, utiliser des valeurs par défaut
        if 'in_channels' not in locals() or in_channels is None:
            in_channels = 6  # RGB (3) + 3 canaux multispectraux
            st.warning(f"Impossible de détecter le nombre de canaux, utilisation par défaut: {in_channels}")
        
        if 'num_classes' not in locals() or num_classes is None:
            num_classes = len(CLASSES)
            st.warning(f"Impossible de détecter le nombre de classes, utilisation par défaut: {num_classes}")
        
        # Créer le modèle
        model = DenseNetMultimodal(in_channels=in_channels, num_classes=num_classes)
        
        # Charger les poids
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Mettre en mode évaluation
        model.eval()
        
        st.info(f"DenseNet chargé avec {in_channels} canaux d'entrée et {num_classes} classes")
        return model
        
    except Exception as e:
        import traceback
        st.error(f"Erreur lors du chargement du modèle DenseNet: {e}")
        st.error(f"Détails: {traceback.format_exc()}")
        return None

def preprocess_early_fusion(rgb_image, ms_images, ms_channels=3):
    """
    Prétraite une image RGB et un ensemble d'images multispectrales pour l'entrée du modèle DenseNet
    
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
    
    st.info(f"Tensor préparé pour DenseNet: {input_tensor.shape}")
    return input_tensor

def predict_with_densenet(model, input_tensor):

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    probabilities = probs.squeeze().numpy()
    
    return CLASSES[pred_class], {cls: float(prob) for cls, prob in zip(CLASSES, probabilities)}