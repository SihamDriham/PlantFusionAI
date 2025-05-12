# Configuration globale de l'application

# Classes pour la prédiction
CLASSES = ["Healthy", "Armillaria_Stage_1", "Armillaria_Stage_2", "Armillaria_Stage_3"]

# Mapping des noms de classes pour l'affichage
FRIENDLY_CLASS_NAMES = {
    "Healthy": "Healthy",
    "Armillaria_Stage_1": "Armillaria - Stade 1",
    "Armillaria_Stage_2": "Armillaria - Stade 2",
    "Armillaria_Stage_3": "Armillaria - Stade 3"
}

# Chemins des modèles
MODEL_PATHS = {
    "multispectral": "best_custom_cnn_model.h5",
    "rgb_resnet": "cherry_tree_resnet_pytorch.pth",
    "early_fusion": "best_model.pth"
}

# Paramètres de prétraitement d'image
IMAGE_SIZE = (224, 224)  # Taille d'entrée standard pour les modèles

# Paramètres de normalisation pour les modèles PyTorch
PYTORCH_MEAN = [0.485, 0.456, 0.406]
PYTORCH_STD = [0.229, 0.224, 0.225]

# Paramètres d'interface utilisateur
UI_COLORS = {
    "healthy": "#e8f5e9",
    "diseased": "#fbe9e7",
    "healthy_border": "#4CAF50",
    "diseased_border": "#FF5722"
}