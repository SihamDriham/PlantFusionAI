import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import streamlit as st

# Configuration
DESIRED_SIZE = (224, 224)
CLASSES = ['Healthy', 'Armillaria_Stage_1', 'Armillaria_Stage_2', 'Armillaria_Stage_3']

# CNN Model Definition
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(0.1)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MultimodalCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(MultimodalCNN, self).__init__()
        self.rgb_branch = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2)
        )
        
        self.ms_branch = nn.Sequential(
            ConvBlock(in_channels-3, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2)
        )
        
        self.fused_branch = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 256), 
            ConvBlock(256, 512),
            nn.MaxPool2d(2, 2),
            ConvBlock(512, 512),
            ConvBlock(512, 512),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((7, 7)))
        
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 1024),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes))
        
    def forward(self, x):
        rgb_input = x[:, :3]
        ms_input = x[:, 3:]
        rgb_features = self.rgb_branch(rgb_input)
        ms_features = self.ms_branch(ms_input)
        fused = torch.cat((rgb_features, ms_features), dim=1)
        features = self.fused_branch(fused)
        features = torch.flatten(features, 1)
        output = self.classifier(features)
        return output

@st.cache_resource
def load_cnn_multimodal_model(checkpoint_path="models/multimodal/best_model_multimodal_cnn.pth"):
    """
    Charge le modèle CNN multimodal pour la fusion précoce
    """
    try:
        if not os.path.exists(checkpoint_path):
            st.warning(f"Le fichier modèle '{checkpoint_path}' n'existe pas. Utilisation du mode simulation.")
            return None
            
        # Charger le checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Examiner la structure du checkpoint
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Analyser les clés pour déterminer la structure du modèle
        st.info("Clés disponibles dans le state_dict: " + ", ".join(list(state_dict.keys())[:5]) + "...")
        
        # Essayons de déterminer la structure du modèle original
        is_simple_cnn = any('features.0.weight' in k for k in state_dict.keys())
        
        if is_simple_cnn:
            st.info("Détection d'un modèle CNN simple avec structure linéaire")
            # Format simplifié avec features et classifier directs
            num_classes = 4  # Par défaut
            in_channels = 6  # Par défaut
            
            # Essayer de détecter le nombre de classes
            for key in state_dict.keys():
                if key.startswith('classifier') and key.endswith('weight') and len(state_dict[key].shape) == 2:
                    num_classes = state_dict[key].shape[0]
                    break
            
            # Essayer de détecter le nombre de canaux d'entrée
            for key in state_dict.keys():
                if key == 'features.0.weight':
                    in_channels = state_dict[key].shape[1]
                    break
            
            st.info(f"Détection d'un modèle avec {in_channels} canaux et {num_classes} classes")
            
            # Créer un modèle simplifié qui correspond à la structure du state_dict
            class SimpleCNN(nn.Module):
                def __init__(self, in_channels, num_classes):
                    super(SimpleCNN, self).__init__()
                    self.features = nn.Sequential(
                        nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(2, 2),
                        nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1, 1))
                    )
                    self.classifier = nn.Sequential(
                        nn.Linear(128, num_classes)
                    )
                
                def forward(self, x):
                    x = self.features(x)
                    x = torch.flatten(x, 1)
                    x = self.classifier(x)
                    return x
            
            model = SimpleCNN(in_channels=in_channels, num_classes=num_classes)
        else:
            # Essayons la structure MultimodalCNN
            in_channels = 6  # Par défaut
            num_classes = len(CLASSES)
            st.info(f"Utilisation du modèle MultimodalCNN avec {in_channels} canaux et {num_classes} classes")
            model = MultimodalCNN(in_channels=in_channels, num_classes=num_classes)
        
        # Essayons de charger les poids, en ignorant les incompatibilités
        try:
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
            st.success("Modèle chargé avec succès (certains poids peuvent avoir été ignorés)")
        except Exception as e:
            st.warning(f"Chargement partiel des poids: {str(e)}")
        
        # Mettre en mode évaluation
        model.eval()
        return model
        
    except Exception as e:
        import traceback
        st.error(f"Erreur lors du chargement du modèle CNN: {e}")
        st.error(f"Détails: {traceback.format_exc()}")
        return None

def preprocess_multimodal(rgb_image, ms_images, ms_channels=3):
    """
    Prétraite une image RGB et un ensemble d'images multispectrales pour l'entrée du modèle CNN
    
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
    
    st.info(f"Tensor préparé pour CNN: {input_tensor.shape}")
    return input_tensor

def predict_with_cnn(model, input_tensor):
    """
    Fait une prédiction avec le modèle CNN
    """
    try:
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        
        probabilities = probs.squeeze().numpy()
        
        # S'assurer que nous avons le bon nombre de classes
        if len(probabilities) != len(CLASSES):
            st.warning(f"Attention: Le modèle produit {len(probabilities)} classes, mais nous attendions {len(CLASSES)} classes.")
            # Gérer la différence de classes
            if len(probabilities) < len(CLASSES):
                # Ajouter des zéros pour les classes manquantes
                padded_probs = np.zeros(len(CLASSES))
                padded_probs[:len(probabilities)] = probabilities
                probabilities = padded_probs
            else:
                # Tronquer aux classes attendues
                probabilities = probabilities[:len(CLASSES)]
            
            # Recalculer la classe prédite
            pred_class = np.argmax(probabilities)
        
        return CLASSES[pred_class], {cls: float(prob) for cls, prob in zip(CLASSES, probabilities)}
    except Exception as e:
        st.error(f"Erreur lors de la prédiction: {str(e)}")
        # Retourner une prédiction par défaut
        return CLASSES[0], {cls: 1.0 if cls == CLASSES[0] else 0.0 for cls in CLASSES}

