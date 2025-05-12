import numpy as np
import cv2
import streamlit as st
import torchvision.transforms as transforms
from PIL import Image
import os
import re
from config.settings import IMAGE_SIZE, PYTORCH_MEAN, PYTORCH_STD

# components/preprocessing.py (ajouter cette fonction)

def preprocess_multispectral_folder(folder_path):
    """
    Prétraite un dossier d'images multispectrales pour l'entrée du modèle
    
    Args:
        folder_path: Chemin vers le dossier contenant les images multispectrales
        
    Returns:
        Données prétraitées pour le modèle
    """
    try:
        # Lister tous les fichiers dans le dossier
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        files = sorted(files)  # Trier les fichiers pour cohérence
        
        if not files:
            st.warning("Aucun fichier trouvé dans le dossier multispectral")
            return None
        
        # Charger et prétraiter chaque image
        ms_images = []
        for file in files:
            file_path = os.path.join(folder_path, file)
            
            # Charger l'image
            try:
                # Essayer avec PIL d'abord
                img = Image.open(file_path)
                img_array = np.array(img)
            except:
                # Si échec, essayer avec OpenCV
                img_array = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
                if img_array is None:
                    st.warning(f"Impossible de charger l'image: {file}")
                    continue
            
            # Redimensionner
            img_resized = cv2.resize(img_array, IMAGE_SIZE)
            
            # Convertir en niveaux de gris si l'image est en couleur
            if len(img_resized.shape) == 3 and img_resized.shape[2] > 1:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Normaliser
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Ajouter à la liste
            ms_images.append(img_normalized)
        
        # Créer un tableau 3D où chaque canal est une image multispectrale
        ms_stack = np.stack(ms_images, axis=-1)
        
        # Ajouter la dimension batch
        ms_batch = np.expand_dims(ms_stack, axis=0)
        
        st.info(f"Forme des données multispectrales: {ms_batch.shape}")
        
        return ms_batch
        
    except Exception as e:
        st.error(f"Erreur lors du prétraitement du dossier multispectral: {e}")
        st.error(f"Détails: {traceback.format_exc()}")
        return None

def preprocess_image(image, model_type, architecture_type):
    # Conversion en tableau numpy
    img_array = np.array(image)
    
    # Redimensionnement à la taille attendue par le modèle
    img_resized = cv2.resize(img_array, IMAGE_SIZE)
    
    # Vérifier le nombre de canaux et adapter si nécessaire
    if len(img_resized.shape) < 3:
        # Convertir une image en niveaux de gris en RGB
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    
    # Prétraitement spécifique pour les images multispectrales
    if model_type == "Multispectrale" or architecture_type == "Multimodale":
        # Pour le modèle multispectral, assurons-nous d'avoir 4 canaux (RGB + NIR par exemple)
        if len(img_resized.shape) == 3:
            # Si l'image est RGB, créer un 4ème canal simulé (NIR)
            if img_resized.shape[2] == 3:
                # On peut simuler un canal NIR en combinant les canaux RGB
                # Par exemple, une simple moyenne des canaux
                nir_channel = np.mean(img_resized, axis=2, keepdims=True)
                img_resized = np.concatenate([img_resized, nir_channel], axis=2)
                st.info("Canal NIR simulé ajouté à l'image RGB")
            # Si l'image a plus que 4 canaux, on garde les 4 premiers
            elif img_resized.shape[2] > 4:
                img_resized = img_resized[:, :, :4]
                st.info(f"Image réduite à 4 canaux pour l'analyse multispectrale")
            # Si l'image a moins de 4 canaux mais plus que RGB, on complète
            elif img_resized.shape[2] < 4:
                # Créer un tableau avec 4 canaux
                temp_img = np.zeros((224, 224, 4), dtype=img_resized.dtype)
                # Copier les canaux existants
                temp_img[:, :, :img_resized.shape[2]] = img_resized
                # Dupliquer le dernier canal pour les canaux manquants
                for i in range(img_resized.shape[2], 4):
                    temp_img[:, :, i] = img_resized[:, :, img_resized.shape[2]-1]
                img_resized = temp_img
                st.info(f"Ajout de canaux pour obtenir 4 canaux au total")
    
    # Normalisation adaptée au modèle (valeurs entre 0 et 1)
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    # Ajout de la dimension batch pour le modèle
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    # Afficher la forme de l'image prétraitée pour déboguer
    st.info(f"Forme de l'image après prétraitement: {img_batch.shape}")
    
    return img_batch

def preprocess_image_pytorch(image):

    # Définir les transformations pour ResNet
    preprocess = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=PYTORCH_MEAN, std=PYTORCH_STD),
    ])
    
    # Appliquer les transformations
    input_tensor = preprocess(image)
    
    # Ajouter la dimension batch
    input_batch = input_tensor.unsqueeze(0)
    
    st.info(f"Image prétraitée pour PyTorch: {input_batch.shape}")
    
    return input_batch

def preprocess_multispectral_folder(folder_path):

    # Liste des fichiers dans le dossier
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Identifier les bandes spectrales à partir des noms de fichiers
    # Exemple: rouge.jpg, bleu.jpg, vert.jpg, nir.jpg
    bands = {}
    band_patterns = {
        'rouge': r'(rouge|red)',
        'vert': r'(vert|green)',
        'bleu': r'(bleu|blue)',
        'nir': r'(nir|proche_infrarouge|near_infrared)'
    }
    
    # Chercher les fichiers correspondant à chaque bande
    for file in files:
        file_lower = file.lower()
        for band_name, pattern in band_patterns.items():
            if re.search(pattern, file_lower):
                bands[band_name] = os.path.join(folder_path, file)
                break
    
    # Vérifier que nous avons au moins les bandes importantes
    required_bands = ['rouge', 'vert', 'bleu']
    missing_bands = [band for band in required_bands if band not in bands]
    
    if missing_bands:
        st.warning(f"Bandes manquantes dans le dossier: {', '.join(missing_bands)}")
        st.info("Utilisation de canaux simulés pour les bandes manquantes")
    
    # Charger et prétraiter chaque bande
    processed_bands = {}
    for band_name, file_path in bands.items():
        # Charger l'image
        img = Image.open(file_path)
        # Convertir en tableau numpy
        img_array = np.array(img)
        # Redimensionnement
        img_resized = cv2.resize(img_array, IMAGE_SIZE)
        # Convertir en niveaux de gris si nécessaire
        if len(img_resized.shape) == 3:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        # Normalisation
        img_normalized = img_resized.astype(np.float32) / 255.0
        # Stocker la bande prétraitée
        processed_bands[band_name] = img_normalized
    
    # Créer des bandes simulées pour les bandes manquantes
    if 'rouge' not in processed_bands and 'vert' in processed_bands:
        processed_bands['rouge'] = processed_bands['vert'] * 1.2
        st.info("Bande rouge simulée à partir de la bande verte")
    if 'vert' not in processed_bands and 'rouge' in processed_bands:
        processed_bands['vert'] = processed_bands['rouge'] * 0.8
        st.info("Bande verte simulée à partir de la bande rouge")
    if 'bleu' not in processed_bands and 'vert' in processed_bands:
        processed_bands['bleu'] = processed_bands['vert'] * 0.7
        st.info("Bande bleue simulée à partir de la bande verte")
    if 'nir' not in processed_bands:
        # Simuler NIR en combinant les autres bandes si disponibles
        if 'rouge' in processed_bands and 'vert' in processed_bands:
            processed_bands['nir'] = (processed_bands['rouge'] + processed_bands['vert']) / 2
            st.info("Bande NIR simulée à partir des bandes rouge et verte")
        elif 'rouge' in processed_bands:
            processed_bands['nir'] = processed_bands['rouge'] * 1.3
            st.info("Bande NIR simulée à partir de la bande rouge")
        elif 'vert' in processed_bands:
            processed_bands['nir'] = processed_bands['vert'] * 1.4
            st.info("Bande NIR simulée à partir de la bande verte")
    
    # Créer la matrice multispectrale combinée
    height, width = IMAGE_SIZE
    combined_image = np.zeros((height, width, 4), dtype=np.float32)
    
    # Remplir les canaux dans l'ordre: R, G, B, NIR
    if 'rouge' in processed_bands:
        combined_image[:, :, 0] = processed_bands['rouge']
    if 'vert' in processed_bands:
        combined_image[:, :, 1] = processed_bands['vert']
    if 'bleu' in processed_bands:
        combined_image[:, :, 2] = processed_bands['bleu']
    if 'nir' in processed_bands:
        combined_image[:, :, 3] = processed_bands['nir']
    
    # Ajout de la dimension batch pour le modèle
    combined_image_batch = np.expand_dims(combined_image, axis=0)
    
    st.info(f"Forme de l'image multispectrale après prétraitement: {combined_image_batch.shape}")
    
    return combined_image_batch