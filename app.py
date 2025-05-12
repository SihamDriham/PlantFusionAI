import streamlit as st
import time
import os
import traceback
from PIL import Image
import tempfile

from models.model_loader import load_multispectral_model, load_rgb_model, load_multimodal_early_fusion_model
from components.preprocessing import preprocess_image, preprocess_image_pytorch, preprocess_multispectral_folder
from components.ui import info_box, display_result, display_fusion_visualization
from components.prediction import classify_with_model, classify_with_pytorch_model, simulate_classification
from config.settings import CLASSES

# Configuration de la page avec thème et mise en page
st.set_page_config(
    page_title="Analyse de Santé des Plantes",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Appliquer des styles CSS personnalisés
# Lire le contenu du fichier CSS
with open("assets/style.css") as f:
    css = f"<style>{f.read()}</style>"
    st.markdown(css, unsafe_allow_html=True)

def main():
    # Barre latérale - Logo et titre
    with st.sidebar:
        st.markdown('<p class="logo-text">🌿 PlantVision AI</p>', unsafe_allow_html=True)
        st.markdown("---")
        st.subheader("Configuration du modèle")
    
    # Titre principal avec icône
    st.title("🌿 Système d'Analyse de Santé des Plantes")
    st.write("Chargez une image ou un dossier d'images de plante pour détecter si elle est saine ou malade.")
    
    # Sidebar pour les options
    with st.sidebar:
        # Choix du type d'architecture
        architecture_type = st.selectbox(
            "Type d'architecture:",
            ["Unimodale", "Multimodale"]
        )
        
        # Options selon le type sélectionné
        if architecture_type == "Unimodale":
            model_type = st.selectbox(
                "Type d'image:",
                ["RGB", "Multispectrale"]
            )
            
            if model_type == "Multispectrale":
                multispectral_approach = st.selectbox(
                    "Approche du modèle multispectral:",
                    ["Cnn","lightweight", "mobileNetV3Small", "EfficientNetB0", "Densenet121", "Resnet18"]
                )
            elif model_type == "RGB":
                # Nouveau sélecteur pour les approches RGB
                rgb_approach = st.selectbox(
                    "Approche du modèle RGB:",
                    ["ResNet50", "ResNet18", "EfficientNetB0"]
                )
            
            selected_model = f"Unimodale - {model_type}"

        else:  # Multimodale
            model_type = st.selectbox(
                "Type de fusion:",
                ["Early Fusion", "Hybrid Fusion", "Late Fusion"]
            )
            selected_model = f"Multimodale - {model_type}"
            
            # Ajout du choix d'approche pour Early Fusion
            if model_type == "Early Fusion":
                fusion_approach = st.selectbox(
                    "Approche de fusion:",
                    ["Cnn", "EfficientNet", "ResNet", "Xception", "DenseNet"]
                )
        
        # Affichage de l'architecture sélectionnée
        st.markdown("---")
        st.markdown(f"**Architecture sélectionnée:**")
        
        # Afficher l'architecture et l'approche sélectionnées
        if selected_model == "Multimodale - Early Fusion":
            st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 5px solid #4CAF50;">
                    <p style="margin: 0; font-weight: bold;">{selected_model} - {fusion_approach}</p>
                </div>
            """, unsafe_allow_html=True)
        elif selected_model == "Unimodale - Multispectrale":
            st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 5px solid #4CAF50;">
                    <p style="margin: 0; font-weight: bold;">{selected_model} - {multispectral_approach}</p>
                </div>
            """, unsafe_allow_html=True)
        elif selected_model == "Unimodale - RGB":
            st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 5px solid #4CAF50;">
                    <p style="margin: 0; font-weight: bold;">{selected_model} - {rgb_approach}</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div style="background-color: #e8f5e9; padding: 10px; border-radius: 5px; border-left: 5px solid #4CAF50;">
                    <p style="margin: 0; font-weight: bold;">{selected_model}</p>
                </div>
            """, unsafe_allow_html=True)

        
        # Chargement du modèle approprié selon la sélection
        if selected_model == "Unimodale - Multispectrale":
            multispectral_model = load_multispectral_model(multispectral_approach.lower())

            if multispectral_model is not None:
                st.success("✅ Modèle multispectral chargé avec succès!")
            else:
                st.error("❌ Échec du chargement du modèle multispectral.")
                
        elif selected_model == "Unimodale - RGB":
            # Charger le modèle RGB avec l'approche sélectionnée
            rgb_model = load_rgb_model(rgb_approach.lower())
            if rgb_model is not None:
                st.success(f"✅ Modèle RGB {rgb_approach} chargé avec succès!")
            else:
                st.error(f"❌ Échec du chargement du modèle RGB {rgb_approach}.")
                
        elif selected_model == "Multimodale - Early Fusion":
            # Charger le modèle de fusion précoce avec l'approche sélectionnée
            early_fusion_model = load_multimodal_early_fusion_model(fusion_approach.lower())
            if early_fusion_model is not None:
                st.success(f"✅ Modèle de fusion précoce ({fusion_approach}) chargé avec succès!")
            else:
                st.error(f"❌ Échec du chargement du modèle de fusion précoce ({fusion_approach}).")
        else:
            # Pour les autres options, on simule seulement
            st.info("Cette configuration utilisera un modèle simulé pour la démonstration.")
            
        st.markdown("---")
        st.caption("© 2025 PlantVision AI")
    
    # Configuration particulière pour Early Fusion qui nécessite une image RGB et des images multispectrales
    if selected_model == "Multimodale - Early Fusion":
        st.info("Pour la fusion précoce, veuillez charger à la fois une image RGB et un dossier d'images multispectrales.")
        
        # Créer des colonnes pour les différents chargements
        col_rgb, col_multi = st.columns(2)
        
        with col_rgb:
            st.subheader("1. Image RGB")
            uploaded_file_rgb = st.file_uploader("Chargez l'image RGB...", 
                                           type=["jpg", "jpeg", "png"], 
                                           key="rgb_uploader")
            
            if uploaded_file_rgb:
                st.image(Image.open(uploaded_file_rgb), caption="Image RGB", width=250)
        
        with col_multi:
            st.subheader("2. Images multispectrales")
            uploaded_folder = st.file_uploader("Chargez les images multispectrales...", 
                                         type=["jpg", "jpeg", "png", "tif"], 
                                         accept_multiple_files=True,
                                         key="multispectral_uploader")
            
            if uploaded_folder and len(uploaded_folder) > 0:
                st.write(f"{len(uploaded_folder)} images multispectrales chargées")
                # Afficher un aperçu des premières images
                if len(uploaded_folder) > 0:
                    preview_cols = st.columns(min(3, len(uploaded_folder)))
                    for i, file in enumerate(uploaded_folder[:3]):
                        with preview_cols[i]:
                            st.image(Image.open(file), caption=f"Image {i+1}", width=120)
        
        if uploaded_file_rgb and uploaded_folder and len(uploaded_folder) > 0:
            st.markdown("---")
            
            # Bouton pour lancer l'analyse avec l'approche spécifiée
            if st.button(f"Lancer l'analyse avec fusion précoce ({fusion_approach})", type="primary"):
                # Créer deux colonnes pour l'affichage des informations et des résultats
                result_col1, result_col2 = st.columns(2)
                
                with result_col1:
                    st.subheader("Images d'entrée")
                    # Afficher l'image RGB
                    st.write("**Image RGB:**")
                    rgb_image = Image.open(uploaded_file_rgb)
                    st.image(rgb_image, caption="Image RGB", width=300)
                    
                    # Afficher un aperçu des images multispectrales
                    st.write("**Images multispectrales:**")
                    multi_cols = st.columns(min(4, len(uploaded_folder)))
                    for i, file in enumerate(uploaded_folder[:4]):
                        with multi_cols[i % 4]:
                            multi_image = Image.open(file)
                            st.image(multi_image, caption=f"Band {i+1}", width=150)
                
                with result_col2:
                    st.subheader("Résultat de l'analyse")
                    
                    # Afficher une barre de progression
                    progress_bar = st.progress(0)
                    
                    # Analyser les images avec une progression
                    with st.spinner(f"Analyse de fusion précoce ({fusion_approach}) en cours..."):
                        # Créer un dossier temporaire pour stocker les fichiers multispectraux
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Enregistrer tous les fichiers dans le dossier temporaire
                            file_paths = []
                            for uploaded_file in uploaded_folder:
                                file_path = os.path.join(temp_dir, uploaded_file.name)
                                with open(file_path, "wb") as f:
                                    f.write(uploaded_file.getbuffer())
                                file_paths.append(file_path)
                            
                            # Liste des images multispectrales (PIL)
                            ms_images = [Image.open(file) for file in uploaded_folder]
                            
                            # Simulation de progression
                            for i in range(100):
                                time.sleep(0.01)
                                progress_bar.progress(i + 1)
                            
                            # Cas spécifique pour DenseNet
                            if fusion_approach.lower() == "densenet" and 'early_fusion_model' in locals() and early_fusion_model is not None:
                                from components.prediction import classify_with_densenet_model
                                result, confidence = classify_with_densenet_model(
                                    rgb_image, 
                                    ms_images, 
                                    early_fusion_model
                                )
                            elif fusion_approach.lower() == "efficientnet" and 'early_fusion_model' in locals() and early_fusion_model is not None:
                                from components.prediction import classify_with_efficientnet_model
                                result, confidence = classify_with_efficientnet_model(
                                    rgb_image, 
                                    ms_images, 
                                    early_fusion_model
                                )
                            elif fusion_approach.lower() == "resnet" and 'early_fusion_model' in locals() and early_fusion_model is not None:
                                from components.prediction import classify_with_resnet_model
                                result, confidence = classify_with_resnet_model(
                                    rgb_image, 
                                    ms_images, 
                                    early_fusion_model
                                )
                            elif fusion_approach.lower() == "xception" and 'early_fusion_model' in locals() and early_fusion_model is not None:
                                from components.prediction import classify_with_xception_model
                                result, confidence = classify_with_xception_model(
                                    rgb_image, 
                                    ms_images, 
                                    early_fusion_model
                                )
                            elif fusion_approach.lower() == "cnn" and 'early_fusion_model' in locals() and early_fusion_model is not None:
                                from components.prediction import classify_with_cnn_model
                                result, confidence = classify_with_cnn_model(
                                    rgb_image, 
                                    ms_images, 
                                    early_fusion_model
                                )
                            else:
                                # Prétraiter l'image RGB
                                rgb_processed = preprocess_image_pytorch(Image.open(uploaded_file_rgb))
                                
                                # Prétraiter les images multispectrales
                                multi_processed = preprocess_multispectral_folder(temp_dir)
                                
                                # Exécuter l'inférence avec le modèle early fusion
                                if 'early_fusion_model' in locals() and early_fusion_model is not None:
                                    result, confidence = classify_with_model([rgb_processed, multi_processed], early_fusion_model)
                                else:
                                    # Utiliser la classification simulée
                                    result, confidence = simulate_classification()
                            
                            # Affichage du résultat
                            display_result(result, confidence)
                            
                            # Afficher la visualisation de fusion
                            # st.write("**Visualisation de la fusion:**")
                            # display_fusion_visualization(f"Early Fusion - {fusion_approach}")

    # Option de chargement pour les modèles multispectraux (sauf Early Fusion qui est traité séparément)
    elif selected_model == "Unimodale - Multispectrale" or (selected_model.startswith("Multimodale") and "Early Fusion" not in selected_model):
        st.info("Pour l'analyse multispectrale, veuillez charger un dossier contenant les images multispectrales.")
        uploaded_folder = st.file_uploader("Chargez un dossier d'images multispectrales...", 
                                         type=["jpg", "jpeg", "png", "tif"], 
                                         accept_multiple_files=True)
        
        if uploaded_folder and len(uploaded_folder) > 0:
            # Créer un dossier temporaire pour stocker les fichiers
            with tempfile.TemporaryDirectory() as temp_dir:
                # Enregistrer tous les fichiers dans le dossier temporaire
                file_paths = []
                for uploaded_file in uploaded_folder:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(file_path)
                
                # Afficher un séparateur pour une meilleure organisation
                st.markdown("---")
                
                # Créer deux colonnes pour l'affichage des images et des résultats
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Images chargées")
                    # Afficher un aperçu des images (maximum 4)
                    images_grid = st.columns(min(4, len(uploaded_folder)))
                    for i, uploaded_file in enumerate(uploaded_folder[:4]):
                        with images_grid[i % 4]:
                            image = Image.open(uploaded_file)
                            st.image(image, caption=f"Image {i+1}", use_column_width=True)
                    
                    # Afficher les informations sur le dossier
                    st.markdown(f"""
                        <div style="font-size: 0.9rem; color: #666;">
                            <p>Nombre d'images chargées: {len(uploaded_folder)}</p>
                            <p>Types de fichiers: {', '.join(set([f.type for f in uploaded_folder]))}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.subheader("Résultat de l'analyse")
                    
                    # Afficher une barre de progression
                    progress_bar = st.progress(0)
                    
                    # Analyser les images avec une progression
                    with st.spinner("Analyse multispectrale en cours..."):
                        # Simulation de progression
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        # Sélection et exécution du modèle approprié
                        if selected_model == "Unimodale - Multispectrale" and 'multispectral_model' in locals() and multispectral_model is not None:
                            # Traitement du dossier d'images multispectrales
                            processed_images = preprocess_multispectral_folder(temp_dir)
                            result, confidence = classify_with_model(processed_images, multispectral_model)
                            
                        else:
                            # Utiliser la classification simulée pour les autres cas
                            processed_images = preprocess_multispectral_folder(temp_dir)
                            result, confidence = simulate_classification()
                        
                        # Affichage du résultat
                        display_result(result, confidence)
                        
                        # Afficher des visualisations selon le modèle
                        if architecture_type == "Multimodale":
                            st.write("**Visualisation de la fusion:**")
                            display_fusion_visualization(model_type)
    else:
        # Pour les modèles RGB, conserver le comportement original (une seule image)
        uploaded_file = st.file_uploader("Chargez une image de plante...", type=["jpg", "jpeg", "png", "tif"])
        
        if uploaded_file is not None:
            # Afficher un séparateur pour une meilleure organisation
            st.markdown("---")
            
            # Créer deux colonnes pour l'affichage de l'image et des résultats
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Image de la plante")
                image = Image.open(uploaded_file)
                st.image(image, caption="Image chargée", use_column_width=True)
                
                # Afficher les informations de l'image
                import numpy as np
                img_array = np.array(image)
                st.markdown(f"""
                    <div style="font-size: 0.9rem; color: #666;">
                        <p>Dimensions: {img_array.shape[1]} x {img_array.shape[0]} pixels</p>
                        <p>Format: {uploaded_file.type}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Résultat de l'analyse")
                
                # Afficher une barre de progression
                progress_bar = st.progress(0)
                
                # Analyser l'image avec une progression
                with st.spinner(f"Analyse avec {rgb_approach} en cours..."):
                    # Simulation de progression
                    for i in range(100):
                        time.sleep(0.01)
                        progress_bar.progress(i + 1)
                    
                    # Sélection et exécution du modèle approprié
                    if selected_model == "Unimodale - RGB" and 'rgb_model' in locals() and rgb_model is not None:
                        processed_image = preprocess_image_pytorch(image)
                        result, confidence = classify_with_pytorch_model(processed_image, rgb_model)
                    else:
                        # Utiliser la classification simulée pour les autres cas
                        processed_image = preprocess_image(image, model_type, architecture_type)
                        result, confidence = simulate_classification()
                    
                    # Affichage du résultat
                    display_result(result, confidence)
    
    # Ajouter un pied de page
    st.markdown("""
        <div class="footer">
            <p>Développé pour l'analyse de santé des plantes | Technologie basée sur l'IA</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()