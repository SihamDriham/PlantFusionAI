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

def classify_with_xception_model(rgb_image, ms_images, model):
    """
    Classifie une image à l'aide du modèle Xception multimodal
    
    Args:
        rgb_image: Image RGB (PIL Image)
        ms_images: Liste d'images multispectrales (liste de PIL Image)
        model: Modèle Xception chargé
        
    Returns:
        classe_prédite, dictionnaire de confiance
    """
    from components.xception_model import preprocess_early_fusion, predict_with_xception
    
    # Déterminer le nombre de canaux multispectraux attendus
    # Selon la structure du modèle, nous pouvons déduire le nombre de canaux MS
    in_channels = model.conv1.weight.shape[1]
    ms_channels = in_channels - 3  # Soustraire les 3 canaux RGB
    
    # Prétraiter les images
    input_tensor = preprocess_early_fusion(rgb_image, ms_images, ms_channels)
    
    # Faire la prédiction
    result, confidence_dict = predict_with_xception(model, input_tensor)
    
    # Pour la compatibilité avec l'interface existante
    confidence = max(confidence_dict.values()) * 100  # Convertir en pourcentage
    
    return result, confidence

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        
        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x

class XceptionMultimodal(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(XceptionMultimodal, self).__init__()
        
        # Entrée adaptée pour le nombre de canaux d'entrée
        self.conv1 = nn.Conv2d(in_channels, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        
        # Entry flow
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        
        # Middle flow
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        
        # Exit flow
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2048, num_classes)
        
        # Initialisation des poids
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Middle flow
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        
        # Exit flow
        x = self.block12(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

@st.cache_resource
def load_xception_multimodal_model(checkpoint_path="models/multimodal/best_model_xception.pth"):
    """
    Charge le modèle Xception multimodal pour la fusion précoce
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
            in_channels = state_dict['conv1.weight'].shape[1]
            num_classes = state_dict['fc.weight'].shape[0]
        else:
            # Format avec le modèle complet
            state_dict = checkpoint
            # Essayer de détecter les dimensions à partir des clés du state_dict
            for key in state_dict.keys():
                if 'conv1.weight' in key:
                    in_channels = state_dict[key].shape[1]
                    break
            
            # Détecter le nombre de classes
            for key in state_dict.keys():
                if 'fc.weight' in key:
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
        model = XceptionMultimodal(in_channels=in_channels, num_classes=num_classes)
        
        # Charger les poids
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Mettre en mode évaluation
        model.eval()
        
        st.info(f"Xception chargé avec {in_channels} canaux d'entrée et {num_classes} classes")
        return model
        
    except Exception as e:
        import traceback
        st.error(f"Erreur lors du chargement du modèle Xception: {e}")
        st.error(f"Détails: {traceback.format_exc()}")
        return None

def preprocess_early_fusion(rgb_image, ms_images, ms_channels=5):
    """
    Prétraite une image RGB et un ensemble d'images multispectrales pour l'entrée du modèle Xception
    
    Args:
        rgb_image: Image RGB au format PIL
        ms_images: Liste d'images multispectrales au format PIL
        ms_channels: Nombre de canaux multispectraux à utiliser
    
    Returns:
        Tensor PyTorch prêt pour l'inférence
    """
    # Taille d'entrée spécifique à Xception
    XCEPTION_SIZE = (299, 299)
    
    # Convertir l'image RGB en tableau numpy et redimensionner
    rgb_img = np.array(rgb_image)
    rgb_img = cv2.resize(rgb_img, XCEPTION_SIZE).astype(np.float32) / 255.0
    
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
        img_array = cv2.resize(img_array, XCEPTION_SIZE)
        
        # Convertir en niveaux de gris si nécessaire
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
        # Normaliser
        img_array = img_array.astype(np.float32) / 255.0
        
        ms_stack.append(img_array)
    
    # Si nous n'avons pas assez d'images MS, compléter avec des zéros
    while len(ms_stack) < ms_channels:
        ms_stack.append(np.zeros(XCEPTION_SIZE, dtype=np.float32))
    
    # Réorganiser les canaux pour PyTorch (C, H, W)
    rgb_channels = rgb_img.transpose(2, 0, 1)
    ms_channels_array = np.stack(ms_stack, axis=0)
    
    # Fusionner RGB et MS
    fused = np.concatenate([rgb_channels, ms_channels_array], axis=0)
    
    # Convertir en tensor PyTorch et ajouter la dimension batch
    input_tensor = torch.from_numpy(fused).float().unsqueeze(0)
    
    st.info(f"Tensor préparé pour Xception: {input_tensor.shape}")
    return input_tensor

def predict_with_xception(model, input_tensor):
    """
    Fait une prédiction avec le modèle Xception
    
    Args:
        model: Modèle Xception chargé
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